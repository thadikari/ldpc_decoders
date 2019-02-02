import numpy as np
import utils
import spa


class Channel:
    def __init__(self, p):
        self.p = p

    def send(self, x):
        return (x + (np.random.random(x.shape) < self.p)) % 2


class SPA:
    def __init__(self, p, code):
        self.spa = spa.SPA(code.parity_mtx)
        self.llr = np.log(1 - p) - np.log(p)

    def decode(self, y):
        priors = self.llr * (1 - 2 * y)
        return self.spa.decode(y, priors)


class ML:
    def __init__(self, p, code):
        self.log_p, self.log_1p = np.log(p), np.log(1 - p)
        self.cb = code.cb

    def decode(self, y):
        num_agrees = np.sum(self.cb == y, axis=1)
        num_diffs = self.cb.shape[1] - num_agrees
        log_prob = num_diffs * self.log_p + num_agrees * self.log_1p
        ind = utils.arg_max_rand(log_prob)
        return self.cb[ind]


class Test(utils.TestCase):
    def test_all(self):
        decoders = [ML, SPA]
        self.sample('4_2_test', 1 / 3, decoders,
                    [1, 1, 0, 1, 1],
                    [1, 0, 0, 1, 1])
        self.sample('7_4_hamming', .1, decoders,
                    [1, 0, 0, 1, 1, 0, 0],
                    [1, 0, 1, 1, 1, 0, 0])
        # .2 will fail as it doesn't have enough
        # strength to change the belief of wrong bit


if __name__ == "__main__":
    import unittest
    import codes

    np.random.seed(0)
    Test().test_all()  # unittest.main()
