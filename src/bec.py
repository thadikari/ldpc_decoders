from scipy.sparse import coo_matrix
import numpy as np

import math_utils as mu
import utils

import lp
import admm


class Channel:
    def __init__(self, p):
        self.p = p

    def send(self, x):
        # any position = 2 means an erasure
        tt = (np.random.random(x.shape) < self.p).astype(int)
        return np.clip(x + tt * 10, 0, 2)


class ML:
    id_keys = []

    def __init__(self, p, _code, **kwargs):
        self.log_p, self.log_1p = np.log(p), np.log(1 - p)
        self.cb = _code.cb
        self.n = self.cb.shape[1]

    def decode(self, y):
        num_erasures = np.sum(y > 1)
        num_agrees = np.sum(self.cb == y, axis=1)
        num_diffs = self.n - num_agrees - num_erasures
        log_prob = num_erasures * self.log_p + num_agrees * self.log_1p
        log_prob[num_diffs > 0] = np.NINF  # CWs that don't match have NINF log likelihood
        ind = mu.arg_max_rand(log_prob)
        return self.cb[ind]


class LLR:
    def __init__(self, dec):
        self.dec, safe_inf = dec, 1e8
        self.llr = np.array([safe_inf, -safe_inf, 0])  # 0 WP1, 1 WP1, 0 OR 1 WP0.5
        if hasattr(self.dec, 'stats'): self.stats = self.dec.stats

    def decode(self, y):
        return self.dec.decode(y, self.llr[y])


class LP(LLR):
    id_keys = lp.LP.id_keys

    def __init__(self, p, _code, **kwargs):
        super().__init__(lp.LP(_code.parity_mtx, **kwargs))


class ADMM(LLR):
    id_keys = admm.ADMM.id_keys

    def __init__(self, p, _code, **kwargs):
        super().__init__(admm.ADMM(_code.parity_mtx, **kwargs))


class ADMMA(LLR):
    id_keys = admm.ADMMA.id_keys

    def __init__(self, p, _code, **kwargs):
        super().__init__(admm.ADMMA(_code.parity_mtx, **kwargs))


class SPA:
    id_keys = ['max_iter']

    def __init__(self, p, _code, **kwargs):
        self.max_iter = kwargs['max_iter']
        self.symbols = np.array([2, 1, 0])  # 2 means erasure
        self.messages = np.array([-1, 1, 0])  # 0 WP1, 1 WP1, 0 OR 1 WP0.5
        self.xx, self.yy = np.where(_code.parity_mtx)

        coo = lambda d_: coo_matrix((d_, (self.xx, self.yy)), shape=_code.parity_mtx.shape)
        self.sum_rows = lambda d_: mu.sum_axis(coo(d_), 1)
        self.sum_cols = lambda d_: mu.sum_axis(coo(d_), 0)

    def decode(self, y):
        xx, yy = self.xx, self.yy
        priors = self.messages[y]
        var_to_chk, chk_to_var = priors[yy] * 1, priors[yy] * 0
        sum_rows, sum_cols = self.sum_rows, self.sum_cols

        x_hat, iter_count = y, 0

        def ret(val):
            # print(val, ':', iter_count)
            return x_hat

        while 1:
            if 0 < self.max_iter <= iter_count: return ret('maximum')
            if np.sum(x_hat == 2) == 0: return ret('decoded')  # no erasures

            # chk_to_var
            sums = sum_rows(1 - np.abs(var_to_chk))  # number of erasures per check
            ma_0, ma_1, ma_2 = (sums == 0)[xx], (sums == 1)[xx], (sums > 1)[xx]

            # checks with no erasures, retain existing certainty for all variables
            # checks with more than 1 erasures, uncertain about all variables
            chk_to_var[ma_0], chk_to_var[ma_2] = var_to_chk[ma_0], 0.

            # erased_pos=0 at 'the' erased position, 1 otherwise
            erased_pos = np.abs(var_to_chk[ma_1])
            # sum up what other vars are suggesting (for bec there cannot be any conflicts)
            incoming = sum_rows(var_to_chk > 0)[xx][ma_1]
            # checks with 1 erasure, uncertain about all variables expect the erased
            chk_to_var[ma_1] = (1 - erased_pos) * (2 * (incoming % 2) - 1)

            # var_to_chk
            marginal = priors + sum_cols(chk_to_var)
            var_to_chk = np.sign(marginal[yy] - chk_to_var, out=var_to_chk)

            # bitwise decoding
            x_new = self.symbols[np.sign(marginal)]
            if (x_hat == x_new).all(): return ret('stopping')  # stopping set
            x_hat = x_new
            iter_count += 1


class MSA(SPA): pass


class Test(utils.TestCase):
    def test_all(self):
        decoders = [ML, LP, SPA, ADMM]
        kwargs = {'max_iter': 100, 'mu': 3., 'eps': 1e-5, 'allow_pseudo': 1}
        self.sample('4_2_test', 1 / 3, decoders,
                    [1, 1, 0, 1, 1],
                    [1, 2, 0, 1, 2],
                    **kwargs)
        self.sample('7_4_hamming', .1, decoders,
                    [1, 0, 0, 1, 1, 0, 0],
                    [2, 0, 2, 1, 1, 0, 2],
                    **kwargs)

    def test_hamming_lp(self):
        decoders = [LP, ADMM]
        kwargs = {'max_iter': 100, 'mu': 3., 'eps': 1e-15, 'allow_pseudo': 1}
        for cw in codes.get_code('7_4_hamming').cb:
            rv = cw.copy()
            rv[-3:] = 2
            self.sample('7_4_hamming', .1, decoders,
                        cw, rv, **kwargs)

    def test_hamming_all(self):
        decoders = [LP]
        # set linprog method='interior-point' to get similar results for SPA and LP
        kwargs = {'max_iter': 100, 'mu': 3., 'eps': 1e-15, 'allow_pseudo': 1}
        errors = sorted(mu.binary_vectors(7), key=lambda k_: k_.sum())
        # np.set_printoptions(linewidth=np.inf), print(np.array(errors).T)
        for cw in codes.get_code('7_4_hamming').cb:
            print(str(cw) + '   ', end='')
            for err in errors:
                ret = self.sample('7_4_hamming', .1, decoders, cw,
                                  np.clip(cw + err * 10, 0, 2),
                                  prt=False, **kwargs)
                print(str(ret[0] + 0) + ' ', end='')
            print('')


if __name__ == "__main__":
    import unittest
    import codes

    np.random.seed(0)
    Test().test_all()  # unittest.main()
