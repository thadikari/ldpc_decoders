import matplotlib.pyplot as plt
import numpy as np
import argparse


def get_gen_mat(omega, n):
    k = len(omega)
    # tri_u = matrix with k ones in kth column (including k=0)
    tri_u = np.zeros([k, k + 1], dtype=int)
    tri_u[:, 1:][np.triu_indices(k)] = 1

    # sample hamming weights of columns of gen_mtx, from 1 to k
    weights = np.random.choice(np.arange(1, len(omega) + 1),
                               n, p=omega)
    gen_mtx = tri_u[:, weights]
    # shuffle columns independently
    gen_mtx = np.apply_along_axis(axis=0, func1d=lambda
        a_: (np.random.shuffle(a_), a_)[1], arr=gen_mtx)

    assert ((gen_mtx.sum(0) == weights).all())
    return gen_mtx


def main(args):
    k, n = 100, 170
    sim_count = 100
    omega = get_soliton(k, 0.1, 0.5)
    arr = [simulate_cw(get_gen_mat(omega, n))
           for _ in range(sim_count)]
    plt.hist(arr)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlim(k, n)
    plt.show()


def simulate_cw(gen_mtx):
    k, n = gen_mtx.shape
    msg = np.random.choice(a=[0, 1], size=k)
    snt = msg @ gen_mtx % 2
    est = np.zeros(k, dtype=int)
    # send symbols till get decoded
    for num_sym in range(k, n + 1):
        # print('num_sym', num_sym)
        rcv = snt[0:num_sym]
        gen_col = gen_mtx[:, :num_sym]
        ret = decode(rcv, gen_col, est)
        # print('rcv:', rcv)
        # print('est:', hist.shape)
        if ret:
            # print('success---------', num_sym)
            return num_sym
        else:
            pass
            # print('retry')
    # print('fail')
    return n  # decoding failure


def decode(rcv, gen_col, est):
    rcv, gen_col = rcv.copy(), gen_col.copy()
    # print('rank', np.linalg.matrix_rank(gen_col))
    while gen_col.sum() != 0:
        # print('----------------------------')
        ripple = gen_col.sum(axis=0) == 1
        if not ripple.any(): return False
        ripple_col = gen_col[:, ripple]
        # print(ripple_col.shape)
        xx, yy = np.nonzero(ripple_col)
        # print(gen_col)
        # print(ripple_col)
        # print('xx, yy', xx, yy)
        # print('ddd', rcv[ripple][yy])
        est[xx] = rcv[ripple][yy]
        rcv += est @ gen_col
        rcv %= 2
        gen_col[xx, :] = 0
        # print(msg)
        # print(rcv)
        # print(gen_col)
    return True


def get_ideal(k):
    rho = np.zeros(k)
    rho[0] = 1 / k
    d = np.arange(2, k + 1)  # +1 to include last element
    rho[d - 1] = 1 / (d * (d - 1))
    return rho


def get_robust(k, c, delta):
    tau = np.zeros(k)
    R = c * np.sqrt(k) * np.log(k / delta)
    ceil = int(np.ceil(k / R))
    d = np.arange(1, ceil - 1 + 1)
    tau[d - 1] = R / (k * d)
    tau[ceil - 1] = np.log(R / delta) * R / k
    return tau


def get_soliton(k, c, delta, plot=False):
    cut = 50
    bar_width = 0.32
    bar_plt = lambda ind, dst, name, clr: \
        plt.bar(np.arange(1, cut + 1) + bar_width * ind,
                dst[:cut], bar_width, linewidth=0,
                label=name, color=clr)

    rho = get_ideal(k)
    tau = get_robust(k, c, delta)
    mu = (rho + tau) / (rho + tau).sum()

    if plot:
        bar_plt(0, rho, 'rho', 'r')
        bar_plt(1, tau, 'tau', 'b')
        bar_plt(2, mu, 'mu', 'y')

        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.show()

    return mu


def test_decoder():
    gen_mtx = np.array([[1, 0, 0], [1, 1, 1],
                        [0, 1, 1], [1, 1, 0]]).T
    msg = np.array([1, 0, 1])
    snt = msg @ gen_mtx % 2
    rcv = snt
    print('message:', msg)
    print('received:', rcv)
    est = np.zeros_like(msg)
    ret = decode(snt, gen_mtx, est)
    print('estimate:', est)


def setup_parser():
    parser = argparse.ArgumentParser()
    return parser


if __name__ == "__main__":
    np.random.seed(2)
    # get_soliton(10000, 0.2, 0.05, True)
    # test_decoder()
    main(setup_parser().parse_args())

    # Usage: python src/luby.py
