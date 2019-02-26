import numpy as np
import math_utils as mu
import utils
import bpa
import lp

noise_var = lambda snr_in_db: 10 ** (-snr_in_db / 10)


class Channel:
    def __init__(self, snr_in_db):
        self.std_dev = np.sqrt(noise_var(snr_in_db))

    def send(self, x):  # incoming cw {0,1}, outgoing cw {-1,+1}
        return (2 * x - 1) + np.random.normal(0, self.std_dev, x.shape)


class BPA:
    def __init__(self, snr_in_db, dec):
        self.noise_var, self.dec = noise_var(snr_in_db), dec

    def decode(self, y):  # incoming cw \reals, outgoing cw {0,1}
        # http://dde.binghamton.edu/filler/mct/lectures/25/mct-lect25-bawgnc.pdf
        return self.dec.decode(y, -2 * y / self.noise_var)


class SPA(BPA):
    def __init__(self, snr_in_db, code, max_iter):
        super().__init__(snr_in_db, bpa.SPA(code.parity_mtx, max_iter))


class MSA(BPA):
    def __init__(self, snr_in_db, code, max_iter):
        super().__init__(snr_in_db, bpa.MSA(code.parity_mtx, max_iter))


class LP(BPA):
    def __init__(self, p, code, max_iter):
        super().__init__(p, lp.LP(code.parity_mtx, max_iter))


class ML:
    def __init__(self, snr_in_db, code, max_iter):
        # map {0,1} to {-1,+1}
        self.cb = code.cb
        self.noise_var = noise_var(snr_in_db)

    def decode(self, y):  # incoming cw \reals, outgoing cw {0,1}
        exponent = -np.square(self.cb * 2 - 1 - y) / (2 * self.noise_var)
        log_prob = np.sum(exponent, axis=1)
        ind = mu.arg_max_rand(log_prob)
        return self.cb[ind]


class Test(utils.TestCase):
    def test_all(self):
        decoders = [ML, SPA, MSA, LP]
        self.sample('4_2_test', 1, decoders, 10,
                    [1, 1, 0, 1, 1],
                    [1, 1, 1.6, .9, 1])
        self.sample('7_4_hamming', .1, decoders, 10,
                    [1, 0, 0, 1, 1, 0, 0],
                    [1, -1, 1.1, 1, 1, -1, -1])


if __name__ == "__main__":
    import unittest
    import codes

    np.random.seed(0)
    Test().test_all()  # unittest.main()
