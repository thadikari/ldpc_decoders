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
    def __init__(self, p, code, max_iter):
        super().__init__(p, bpa.SPA(code.parity_mtx, max_iter))


class MSA(LLR):
    def __init__(self, p, code, max_iter):
        super().__init__(p, bpa.MSA(code.parity_mtx, max_iter))


class LP(LLR):
    def __init__(self, p, code, max_iter):
        super().__init__(p, lp.LP(code.parity_mtx, max_iter))


class ADMM(LLR):
    def __init__(self, p, code, max_iter):
        super().__init__(p, admm.ADMM(code.parity_mtx, max_iter))


class ML:
    def __init__(self, p, code, max_iter):
        self.log_p, self.log_1p = np.log(p), np.log(1 - p)
        self.cb = code.cb

    def decode(self, y):
        num_agrees = np.sum(self.cb == y, axis=1)
        num_diffs = self.cb.shape[1] - num_agrees
        log_prob = num_diffs * self.log_p + num_agrees * self.log_1p
        ind = mu.arg_max_rand(log_prob)
        return self.cb[ind]


class Test(utils.TestCase):
    def test_all(self):
        decoders = [ML, SPA, MSA, LP, ADMM]
        self.sample('4_2_test', 1 / 3, decoders, 10,
                    [1, 1, 0, 1, 1],
                    [1, 0, 0, 1, 1])
        self.sample('7_4_hamming', .1, decoders, 10,
                    [1, 0, 0, 1, 1, 0, 0],
                    [1, 0, 1, 1, 1, 0, 0])
        # .2 will fail as it doesn't have enough
        # strength to change the belief of wrong bit


if __name__ == "__main__":
    import unittest
    import codes

    np.random.seed(0)
    Test().test_all()  # unittest.main()
