from scipy.optimize import linprog
import matplotlib.pyplot as plt
import numpy as np
import logging
import utils


class Dist:
    def __init__(self, name, lambda_p, rho_p, eps_BP):
        self.name, self.thresh = name, eps_BP
        self.lambda_p, self.rho_p = lambda_p, rho_p


eval_ = lambda p_, x_: np.polyval(p_, x_)
avg_deg_inv = lambda p_: eval_(np.polyint(p_), 1)  # 0 to 1 integration
rate__ = lambda lambda_p, rho_p,: 1 - avg_deg_inv(rho_p) / avg_deg_inv(lambda_p)
rate_ = lambda code_: rate__(code_.lambda_p, code_.rho_p)

x1 = np.linspace(0, .6, num=50)
plt.plot(x1, x1, 'k--', linewidth=3)

plot_de_eps = lambda code_, eps_, eps_name='eps', extra=None: \
    plt.plot(x1, eps_ * eval_(code_.lambda_p, 1 - eval_(code_.rho_p, 1 - x1)), linewidth=3,
             label='%s: rate=%g, %s=%g%s' % (code_.name, rate_(code_), eps_name, eps_,
                                             '' if extra is None else ', %s' % extra))

plot_de_thresh = lambda code_: plot_de_eps(code_, code_.thresh, 'eps_BP',
                                           'gap=%g' % (1 - code_.thresh - rate_(code_)))

pol2str = lambda pol: ' + '.join(
    '%sx^%d' % ('%g ' % pol[-ind - 1] if pol[-ind - 1] != 1. else '', ind)
    for ind in range(len(pol)) if pol[-ind - 1] != 0.)


class Solver:
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

    def solve_iter_(self, target_rate, interval):
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
            return self.solve_iter_(target_rate, interval_)


def solve_iter(name, target_rate, rho_p,
               l_max=40, tol_l_max=1e-5,
               tol_iter=1e-8, discrete_count=100):
    lambda_p, eps_BP = Solver(l_max, rho_p, discrete_count, tol_iter) \
        .solve_iter_(target_rate, [0., 1.])

    # get minimum degree l_max
    while lambda_p and lambda_p[0] < tol_l_max:
        lambda_p.pop(0)

    return Dist(name, lambda_p, rho_p, eps_BP)


def solve_r(r, target_rate):
    rho_p = [1] + [0] * (r - 1)
    dist = solve_iter('r=%d' % r, target_rate, rho_p)
    print('r = %d' % r)
    print('lambda(x) = %s' % pol2str(dist.lambda_p))
    print('rho(x) = %s' % pol2str(dist.rho_p))
    plot_de_thresh(dist)


def main():
    for r in [6, 5, 4]: solve_r(r, .5)

    if 0:
        opt_mct = Dist('Opt-MCT',
                       [0.1151, 0.1971, 0, 0, 0.0768, 0.202, 0.409, 0],
                       [1, 0, 0, 0, 0, 0], 0.4810)
        # Optimal half-rate code for ?(x) = x**5 and lmax = 8, Modern Coding Theory (MCT) pp.115
        # optimal in the sense this code has the minimum gap to capacity for .5 rate LDPC codes
        print(pol2str(opt_mct.rho_p))
        print(pol2str(opt_mct.lambda_p))
        plot_de_thresh(opt_mct)

        ldpc_36 = Dist('(3,6)', [1, 0, 0], [1, 0, 0, 0, 0, 0], .427)
        # ldpc36 cannot drive error to zero if channel eps=.5 .
        # max channel eps whose err can be driven to zero with ldpc36 is around .427 .
        # epsilon = .3 ==> capacity = 1-.3 = .7 > .5 = rate of ldpc36 ==> this code can drive error to zero
        plot_de_thresh(ldpc_36)
        for eps in [.5, .3]: plot_de_eps(ldpc_36, eps)

    plt.title('Density Evolution for BEC')
    plt.xlabel('$x_l$')
    plt.ylabel('$x_{l+1}$')
    plt.axes().set_aspect('equal')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    utils.setup_console_logger(level=logging.INFO)
    main()
