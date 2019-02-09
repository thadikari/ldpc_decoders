import numpy as np
import utils
import spa


class Channel:
    def __init__(self, p):
        self.p = p

    def send(self, x):
        # any position = 2 means an erasure
        tt = (np.random.random(x.shape) < self.p).astype(int)
        return np.clip(x + tt * 10, 0, 2)


class ML:
    def __init__(self, p, code):
        self.log_p, self.log_1p = np.log(p), np.log(1 - p)
        self.cb = code.cb
        self.n = self.cb.shape[1]

    def decode(self, y):
        num_erasures = np.sum(y > 1)
        num_agrees = np.sum(self.cb == y, axis=1)
        num_diffs = self.n - num_agrees - num_erasures
        log_prob = num_erasures * self.log_p + num_agrees * self.log_1p
        log_prob[num_diffs > 0] = np.NINF  # CWs that don't match have NINF log likelihood
        ind = utils.arg_max_rand(log_prob)
        return self.cb[ind]


class SPA:
    def __init__(self, p, code):
        self.spa = spa.SPA(code.parity_mtx)
        self.llr = np.array([np.inf, -np.inf, 0])

    def decode(self, y):
        return self.spa.decode(y, self.llr[y])


class Test(utils.TestCase):
    def test_all(self):
        decoders = [ML, SPA]
        self.sample('4_2_test', 1 / 3, decoders,
                    [1, 1, 0, 1, 1],
                    [1, 2, 0, 1, 2])
        self.sample('7_4_hamming', .1, decoders,
                    [1, 0, 0, 1, 1, 0, 0],
                    [2, 2, 2, 1, 1, 0, 0])


if __name__ == "__main__":
    import unittest
    import codes

    np.random.seed(0)
    Test().test_all()  # unittest.main()
