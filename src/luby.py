from scipy.sparse import csc_matrix
from multiprocessing import Pool
import numpy as np
import argparse
import logging

import math_utils as mu
import utils


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


def test_sim():
    k, n = 100, 170
    sim_count = 3
    omega = get_soliton(k, 0.1, 0.5)
    arr = [simulate_cw(i, omega, n)[1]
           for i in range(sim_count)]

    print(arr)
    import luby_graph
    luby_graph.plot_hist(arr, k, n, .1)


def simulate_cw(sim_id, omega, n):
    # return sim_id, sim_id + 99
    np.random.seed(sim_id)
    k = len(omega)
    gen_mtx = csc_matrix(get_gen_mat(omega, n))
    msg = np.random.choice(a=[0, 1], size=k)
    snt = (msg @ gen_mtx) % 2
    est = np.zeros(k, dtype=int)
    ret_val = n
    # send symbols till get decoded
    # min_rank = -1
    for num_sym in range(k, n + 1):
        # print('sim_id', sim_id, 'num_sym', num_sym)
        rcv = snt[0:num_sym]
        gen_col = gen_mtx[:, :num_sym]
        ret = decode(rcv, gen_col, est)
        # print('rcv:', rcv)
        # print('est:', hist.shape)
        # if min_rank == -1 and k == np.linalg.matrix_rank(gen_col.toarray()): min_rank = num_sym
        if ret:
            # print('success---------', num_sym, 'min_rank', min_rank)
            # assert ((msg == est).all())
            # print(np.abs(msg - est).sum())
            ret_val = num_sym
            break
        else:
            pass
            # print('retry')
    # print('fail')
    return sim_id, ret_val  # decoding failure


def decode(rcv, gen_col, est):
    csc = csc_matrix(gen_col)
    csc.data = csc.data.copy()
    rcv = rcv.copy()
    while csc.data.sum() != 0:
        ripple = mu.mtx_to_vec(csc.sum(axis=0)) == 1
        if not ripple.any(): return False
        ripple_col = csc[:, ripple]
        xx, yy = np.nonzero(ripple_col)
        est[xx] = rcv[ripple][yy]
        rcv += est @ csc
        rcv %= 2
        csr = csc.tocsr()
        for i in xx: csr.data[csr.indptr[i]:csr.indptr[i + 1]] = 0
        csc = csr.tocsc()
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


def plot_avg_deg(k, delta):
    average = lambda dst: dst @ np.arange(1, len(dst) + 1)
    ll = np.linspace(.01, .1, 50)
    avg_deg = [average(get_soliton(k, c, delta, plot=False)) for c in ll]
    import luby_graph
    luby_graph.plot_avg_deg(ll, avg_deg)


def get_soliton(k, c, delta, plot=False):
    rho = get_ideal(k)
    tau = get_robust(k, c, delta)
    mu = (rho + tau) / (rho + tau).sum()

    if plot:
        import luby_graph
        luby_graph.plot_soliton(rho, tau, mu, c, 103)

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
    parser.add_argument('k', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('c', type=float)
    parser.add_argument('delta', type=float)
    parser.add_argument('count', type=int)
    parser.add_argument('--pool', default=2, type=int)
    return utils.bind_parser_common(parser)


def exec_pool(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.console:
        utils.setup_console_logger(log_level)
    else:
        utils.setup_file_logger(args.data_dir, 'luby', log_level)

    id_keys = ['k', 'n', 'c', 'delta']
    id_val = [str(vars(args)[key]) for key in id_keys]
    saver = utils.Saver(args.data_dir, list(
        zip(['type'] + id_keys, ['luby'] + id_val)))
    log = logging.getLogger('.'.join(id_val))

    k, n, arr = args.k, args.n, []
    omega = get_soliton(k, args.c, args.delta)

    def callback(cb_args):
        sim_id, num_sym = cb_args
        log.info('sim_id=%d, num_sym=%d' % (sim_id, num_sym))
        arr.append(num_sym)
        saver.add_all({'arr': arr})

    pool = Pool(processes=args.pool)
    results = [pool.apply_async(simulate_cw, (x, omega, n,),
                                callback=callback)
               for x in range(args.count)]
    for r in results: r.wait()
    log.info('Finished all!')


if __name__ == "__main__":
    # get_soliton(10000, 0.1, 0.5, True)
    # plot_avg_deg(10000, .5)
    # test_decoder()
    # test_sim()
    exec_pool(setup_parser().parse_args())

    # Usage: python -u src/luby.py 100 170 .1 .5 10
    # Usage: python -u src/luby.py 10000 12000 .1 .5 250 --pool=40 --data-dir=$SCRATCH
