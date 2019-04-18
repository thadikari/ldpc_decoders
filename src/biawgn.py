import numpy as np

import math_utils as mu
import utils

import bpa
import lp
import admm

noise_var = lambda snr_in_db: 10 ** (-snr_in_db / 10)


class Channel:
    def __init__(self, snr_in_db):
        self.std_dev = np.sqrt(noise_var(snr_in_db))

    def send(self, x):  # incoming cw {0,1}, outgoing cw {-1,+1}
        return (2 * x - 1) + np.random.normal(0, self.std_dev, x.shape)


class LLR:
    def __init__(self, snr_in_db, dec):
        self.noise_var, self.dec = noise_var(snr_in_db), dec
        if hasattr(self.dec, 'stats'): self.stats = self.dec.stats

    def decode(self, y):  # incoming cw \reals, outgoing cw {0,1}
        # http://dde.binghamton.edu/filler/mct/lectures/25/mct-lect25-bawgnc.pdf
        return self.dec.decode(y, -2 * y / self.noise_var)


class SPA(LLR):
    id_keys = bpa.SPA.id_keys

    def __init__(self, snr_in_db, _code, **kwargs):
        super().__init__(snr_in_db, bpa.SPA(_code.parity_mtx, **kwargs))


class MSA(LLR):
    id_keys = bpa.MSA.id_keys

    def __init__(self, snr_in_db, _code, **kwargs):
        super().__init__(snr_in_db, bpa.MSA(_code.parity_mtx, **kwargs))


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

    def __init__(self, snr_in_db, _code, **kwargs):
        # map {0,1} to {-1,+1}
        self.cb = _code.cb
        self.noise_var = noise_var(snr_in_db)

    def decode(self, y):  # incoming cw \reals, outgoing cw {0,1}
        exponent = -np.square(self.cb * 2 - 1 - y) / (2 * self.noise_var)
        log_prob = np.sum(exponent, axis=1)
        ind = mu.arg_max_rand(log_prob)
        return self.cb[ind]


class Test(utils.TestCase):
    def test_all(self):
        decoders = [ML, SPA, MSA, LP, ADMM]
        kwargs = {'max_iter': 100, 'mu': 3., 'eps': 1e-5, 'allow_pseudo': 1}
        self.sample('4_2_test', 1, decoders,
                    [1, 1, 0, 1, 1],
                    [1, 1, 1.6, .9, 1],
                    **kwargs)
        self.sample('7_4_hamming', .1, decoders,
                    [1, 0, 0, 1, 1, 0, 0],
                    [1, -1, 1.1, 1, 1, -1, -1],
                    **kwargs)


if __name__ == "__main__":
    import unittest
    import codes

    np.random.seed(0)
    Test().test_all()  # unittest.main()
