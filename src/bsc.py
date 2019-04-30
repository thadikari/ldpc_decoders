import numpy as np

import math_utils as mu
import utils

import bpa
import lp
import admm


class Channel:
    def __init__(self, p):
        self.p = p

    def send(self, x):
        return (x + (np.random.random(x.shape) < self.p)) % 2


class LLR:
    def __init__(self, p, dec):
        self.llr, self.dec = np.log(1 - p) - np.log(p), dec
        if hasattr(self.dec, 'stats'): self.stats = self.dec.stats

    def decode(self, y):
        return self.dec.decode(y, self.llr * (1 - 2 * y))


class SPA(LLR):
    id_keys = bpa.SPA.id_keys

    def __init__(self, p, _code, **kwargs):
        super().__init__(p, bpa.SPA(_code.parity_mtx, **kwargs))


class MSA(LLR):
    id_keys = bpa.MSA.id_keys

    def __init__(self, p, _code, **kwargs):
        super().__init__(p, bpa.MSA(_code.parity_mtx, **kwargs))


class LP(LLR):
    id_keys = lp.LP.id_keys

    def __init__(self, p, _code, **kwargs):
        super().__init__(p, lp.LP(_code.parity_mtx, **kwargs))


class ADMM(LLR):
    id_keys = admm.ADMM.id_keys

    def __init__(self, p, _code, **kwargs):
        super().__init__(p, admm.ADMM(_code.parity_mtx, **kwargs))


class ADMMA(LLR):
    id_keys = admm.ADMMA.id_keys

    def __init__(self, p, _code, **kwargs):
        super().__init__(p, admm.ADMMA(_code.parity_mtx, **kwargs))


class ML:
    id_keys = []

    def __init__(self, p, _code, **kwargs):
        self.log_p, self.log_1p = np.log(p), np.log(1 - p)
        self.cb = _code.cb

    def decode(self, y):
        num_agrees = np.sum(self.cb == y, axis=1)
        num_diffs = self.cb.shape[1] - num_agrees
        log_prob = num_diffs * self.log_p + num_agrees * self.log_1p
        ind = mu.arg_max_rand(log_prob)
        return self.cb[ind]


class Test(utils.TestCase):
    def test_all(self):
        decoders = [ML, SPA, MSA, LP, ADMM]
        kwargs = {'max_iter': 100, 'mu': 3., 'eps': 1e-5, 'allow_pseudo': 1}
        self.sample('4_2_test', 1 / 3, decoders,
                    [1, 1, 0, 1, 1],
                    [1, 0, 0, 1, 1],
                    **kwargs)
        self.sample('7_4_hamming', .1, decoders,
                    [1, 0, 0, 1, 1, 0, 0],
                    [1, 0, 1, 1, 1, 0, 0],
                    **kwargs)
        # .2 will fail as it doesn't have enough
        # strength to change the belief of wrong bit

    def test_LP_vs_ADMM(self):
        decoders = [LP, ADMM]
        kwargs = {'max_iter': -1, 'mu': 3., 'eps': 1e-15, 'allow_pseudo': 1}
        self.sample('7_4_hamming', .1, decoders,
                    [0, 1, 0, 0, 1, 0, 1],
                    [0, 1, 0, 1, 1, 0, 1],
                    **kwargs)

    def test_find_pcws(self):
        md = admm.ADMM
        md = lp.LP
        np.set_printoptions(linewidth=np.inf)
        x = np.array([0, 1, 0, 0, 1, 0, 1])
        y = np.array([0, 1, 0, 1, 1, 0, 1])
        kwargs = {'max_iter': -1, 'mu': 3., 'eps': 1e-5, 'allow_pseudo': 1}
        dec = md(codes.get_code('7_4_hamming').parity_mtx, **kwargs)
        ll = np.copy(x)[np.newaxis, :]
        for i in range(1000):
            z = dec.decode(y, 1 - 2 * y + np.random.rand(7) * 0.001)
            if (np.max(np.abs(ll - z), axis=1) > 1e-3).all():
                ll = np.append(ll, z[np.newaxis, :], axis=0)
                print(z)

    def test_hamming_all(self):
        decoders = [LP]
        # set linprog method='interior-point' to get similar results for SPA and LP
        kwargs = {'max_iter': -1, 'mu': 3., 'eps': 1e-15, 'allow_pseudo': 1}
        errors = sorted(mu.binary_vectors(7), key=lambda k_: k_.sum())
        # np.set_printoptions(linewidth=np.inf), print(np.array(errors).T)
        for cw in codes.get_code('7_4_hamming').cb:
            print(str(cw) + '   ', end='')
            for err in errors:
                ret = self.sample('7_4_hamming', .1, decoders, cw,
                                  (cw + err) % 2,
                                  prt=False, **kwargs)
                print(str(ret[0] + 0) + ' ', end='')
            print('')


if __name__ == "__main__":
    import unittest
    import codes

    np.random.seed(0)
    Test().test_hamming_all()  # unittest.main()
