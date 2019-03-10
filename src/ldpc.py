from scipy.optimize import linprog
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
import argparse
import logging

import utils
import codes


class Dist:  # irregular code distribution
    def __init__(self, name, lambda_p, rho_p, eps_BP):
        self.name, self.thresh = name, eps_BP
        self.lambda_p, self.rho_p = lambda_p, rho_p


eval_ = lambda p_, x_: np.polyval(p_, x_)
avg_deg_inv = lambda p_: eval_(np.polyint(p_), 1)  # 0 to 1 integration
rate__ = lambda lambda_p, rho_p,: 1 - avg_deg_inv(rho_p) / avg_deg_inv(lambda_p)
rate_ = lambda code_: rate__(code_.lambda_p, code_.rho_p)

x1 = np.linspace(0, .6, num=50)
f_eps_x = lambda eps_, lambda_p_, rho_p_, x_: eps_ * eval_(lambda_p_, 1 - eval_(rho_p_, 1 - x_))
f_eps_x1 = lambda eps_, lambda_p_, rho_p_: f_eps_x(eps_, lambda_p_, rho_p_, x1)
plot_de_eps = lambda code_, eps_, eps_name='eps', extra=None: \
    plt.plot(x1, f_eps_x1(eps_, code_.lambda_p, code_.rho_p), linewidth=3,
             label='%s: rate=%g, %s=%g%s' % (code_.name, rate_(code_), eps_name, eps_,
                                             '' if extra is None else ', %s' % extra))

plot_de_thresh = lambda code_: plot_de_eps(code_, code_.thresh, 'eps_BP',
                                           'gap=%g, mul. gap=%g' % (1 - code_.thresh - rate_(code_),
                                                                    1 - rate_(code_) / (1 - code_.thresh)))

pol2str = lambda pol: ' + '.join(
    '%sx^%d' % ('%g ' % pol[-ind - 1] if pol[-ind - 1] != 1. else '', ind)
    for ind in range(len(pol)) if pol[-ind - 1] != 0.)


class LambdaSolver:  # solve for lambda given rho, to reach a target rate
    def __init__(self, l_max, rho_p, discrete_count, tolerance):
        self.log = logging.getLogger('Solver')
        self.rho_p = rho_p
        self.tolerance = tolerance

        # +1 to include l_max
        range_i = np.arange(2, l_max + 1)
        self.c_obj = -1. / range_i
        self.A_eq, self.b_eq = np.ones([1, len(range_i)]), 1.

        # by default, linprog solves for non-negative solutions
        xd = np.linspace(0, 1, num=discrete_count)
        rho1_xd = eval_(rho_p, 1. - xd)
        self.A_ub = (1 - rho1_xd)[:, np.newaxis] ** (range_i - 1)
        self.b_ub = xd

    def solve(self, eps):
        res = linprog(self.c_obj, A_ub=eps * self.A_ub, b_ub=self.b_ub,
                      A_eq=self.A_eq, b_eq=self.b_eq,
                      # options={'disp': True}
                      )
        # return polynomial with highest order coefficient first and lambda_1 = 0
        return np.flip(res.x).tolist() + [0.]

    # Recursively solve for optimal lambda that has target_rate
    # (for BEC) by adjusting channel epsilon
    def solve_lambda_eps_BP_iter(self, target_rate, interval):
        eps = sum(interval) / 2.
        lambda_p = self.solve(eps)
        actual_rate = rate__(lambda_p, self.rho_p)
        self.log.debug('eps=%g, actual_rate=%g' % (eps, actual_rate))

        if abs(actual_rate - target_rate) < self.tolerance:
            self.log.debug('converged!')
            return lambda_p, eps
        else:
            ind = (actual_rate > target_rate) + 0
            interval_ = sorted([interval[ind], eps])
            return self.solve_lambda_eps_BP_iter(target_rate, interval_)


# Solve for optimal lambda distribution and eps_BP given target rate and rho
def solve_dist(name, target_rate, rho_p,
               l_max=40, tol_l_max=1e-5,
               tol_iter=1e-8, discrete_count=100):
    lambda_p, eps_BP = LambdaSolver(l_max, rho_p, discrete_count, tol_iter) \
        .solve_lambda_eps_BP_iter(target_rate, [0., 1.])

    # Remove leading zero coefficients that correspond to
    # higher order terms.
    while lambda_p and lambda_p[0] < tol_l_max:
        lambda_p.pop(0)

    return Dist(name, lambda_p, rho_p, eps_BP)


# Recursively solve for optimal threshold eps_BP
# for a given lambda_p and rho_p
x1_ = np.linspace(1e-3, 1, num=500)
make_dist = lambda name_, lambda_p_, rho_p_: Dist(name_, lambda_p_, rho_p_, solve_eps_BP(lambda_p_, rho_p_))


def solve_eps_BP(lambda_p, rho_p, interval=None, tolerance=1e-8):
    if interval is None: interval = [0., 1]
    eps = sum(interval) / 2.
    gap = max(f_eps_x(eps, lambda_p, rho_p, x1_) - x1_)
    if gap > 0:
        return solve_eps_BP(lambda_p, rho_p, [interval[0], eps])
    elif gap < -tolerance:
        return solve_eps_BP(lambda_p, rho_p, [eps, interval[1]])
    else:
        return eps


# p(x) = x^deg polynomial
reg_pol = lambda deg: [1] + [0] * deg


# Solve, plot, for lambda given a rho with only one term
def solve_plot(rho_r, target_rate):
    dist = solve_dist('rho_r=%d' % rho_r, target_rate, reg_pol(rho_r))
    print('rho_r = %d' % rho_r)
    print('lambda(x) = %s' % pol2str(dist.lambda_p))
    print('rho(x) = %s' % pol2str(dist.rho_p))
    plot_de_thresh(dist)


# Node (var/chk) distribution
# ex: L(x) = \int{polynomial}/int_0_to_1{polynomial}
def get_node_dist(pol):
    int_p = np.polyint(pol)
    return int_p / eval_(int_p, 1)


gen_L_R = lambda code_: (get_node_dist(code_.lambda_p), get_node_dist(code_.rho_p))


def add_sockets(counts, sockets, last):
    for deg in range(len(counts)):
        cnt = counts[-deg - 1]
        # There are [cnt] of degree [deg] nodes, with indices:
        # print(cnt, deg, list(range(last, last + cnt)))
        # print(list(range(last, last + cnt)) * deg)
        sockets.extend(list(range(last, last + cnt)) * deg)
        last += cnt
    return last


def gen_rand_irg_ldpc(args):
    num_var, rho_r, rate = args.len, args.rho, args.rate

    if 1:
        dist = solve_dist('rho_r=%d' % rho_r, rate, reg_pol(rho_r))
        extra = [0, 1, 0, 0, 0, 0, 1, 0, 0]
    else:
        dist = Dist('Opt-MCT',
                    [0.1151, 0.1971, 0, 0, 0.0768, 0.202, 0.409, 0],
                    reg_pol(5), 0.4810)
        extra = [0, 0, 0, 0, 1, 1, 0, 0, 0]

    L_p, _ = gen_L_R(dist)

    sockets_var = []
    last = add_sockets(list(int(it * num_var) for it in L_p),
                       sockets_var, 1)

    # 'extra' must be solved for, not hardcoded.
    # Need to solve for extra s.t. all are integers list
    # Says: add extra[i] number of variables of degree-i to
    # satisfy rem_chk_sock, need_var values
    chk_deg = rho_r + 1
    rem_chk_sock = chk_deg - len(sockets_var) % chk_deg
    need_var = num_var + 1 - last
    print(rem_chk_sock, need_var)

    add_sockets(extra, sockets_var, last)

    num_edges = len(sockets_var)
    assert (num_edges % chk_deg == 0)
    assert (max(sockets_var) == num_var)
    num_chk = int(num_edges / chk_deg)
    sockets_chk = list(range(1, num_chk + 1)) * chk_deg

    for i in range(args.count):
        parity_mtx = np.zeros((num_chk, num_var), int)
        shuffle(sockets_var)
        for chk_ind, var_ind in zip(sockets_chk, sockets_var):
            parity_mtx[chk_ind - 1, var_ind - 1] += 1
        parity_mtx[parity_mtx % 2 == 0] = 0

        code_name = '%d_rho_x%d_rand_ldpc_%d' % (num_var, rho_r, i + 1)
        codes.save_parity_mtx(parity_mtx, code_name)


def plot_density_evolution(args):
    for rho_r in [6, 5, 4]: solve_plot(rho_r, .5)
    # plot_de_thresh(make_dist('Prob 3.4', reg_pol(2), reg_pol(6)))

    if 0:
        opt_mct = make_dist('Opt-MCT', [0.1151, 0.1971, 0, 0, 0.0768, 0.202, 0.409, 0], reg_pol(5))
        # Optimal half-rate code for ?(x) = x**5 and lmax = 8, Modern Coding Theory (MCT) pp.115
        # optimal in the sense this code has the minimum gap to capacity for .5 rate LDPC codes
        print(pol2str(opt_mct.rho_p))
        print(pol2str(opt_mct.lambda_p))
        plot_de_thresh(opt_mct)

        ldpc_36 = make_dist('(3,6)', reg_pol(2), reg_pol(5))
        # ldpc36 cannot drive error to zero if channel eps=.5 .
        # max channel eps whose err can be driven to zero with ldpc36 is around .427 .
        # epsilon = .3 ==> capacity = 1-.3 = .7 > .5 = rate of ldpc36 ==> this code can drive error to zero
        plot_de_thresh(ldpc_36)
        for eps in [.5, .3]: plot_de_eps(ldpc_36, eps)

    plt.plot(x1, x1, 'k--', linewidth=3)
    plt.title('Density Evolution for BEC')
    plt.xlabel('$x_l$')
    plt.ylabel('$x_{l+1}$')
    plt.axes().set_aspect('equal')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='design regular OR irregular code', choices=['plt', 'irg'])
    parser.add_argument('--count', help='number of random codes to generate', type=int)
    parser.add_argument('--len', help='code length', type=int)
    parser.add_argument('--rate', help='design rate', type=float)
    parser.add_argument('--rho', help='regular degree for rho', type=int)
    return parser


def main(args):
    {'plt': plot_density_evolution,
     'irg': gen_rand_irg_ldpc}[args.task](args)


if __name__ == "__main__":
    utils.setup_console_logger(level=logging.INFO)
    main(setup_parser().parse_args())

    # Usage: python src/ldpc.py plt
    # Usage: python src/ldpc.py irg --count=10 --len=1200 --rho=5 --rate=.5
