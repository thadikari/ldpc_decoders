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
        kwargs = {'max_iter': 10, 'mu': 3., 'eps': 1e-5}
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


if __name__ == "__main__":
    import unittest
    import codes

    np.random.seed(0)
    Test().test_all()  # unittest.main()
