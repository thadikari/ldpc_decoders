from scipy.sparse import coo_matrix
import math_utils as mu
import numpy as np


class SPA:
    def __init__(self, parity_mtx, max_iter):
        self.max_iter = max_iter
        self.parity_mtx = parity_mtx
        self.xx, self.yy = np.where(self.parity_mtx)

        coo = lambda d_: coo_matrix((d_, (self.xx, self.yy)), shape=parity_mtx.shape)
        self.prod_rows = lambda d_: mu.prod_nonzero(coo(d_), 1)
        self.sum_cols = lambda d_: mu.sum_axis(coo(d_), 0)

    def decode(self, y, priors):
        xx, yy = self.xx, self.yy
        prod_rows, sum_cols = self.prod_rows, self.sum_cols
        var_to_chk, chk_to_var = priors[yy], None
        x_hat, iter_count = y, 0

        def ret(val):
            # print(val, ':', iter_count)
            return x_hat

        while 1:
            if iter_count >= self.max_iter: return ret('maximum')
            if ((self.parity_mtx @ x_hat) % 2 == 0).all(): return ret('decoded')

            # chk_to_var
            tanned = np.tanh(var_to_chk / 2.)
            chk_msg_prod = prod_rows(tanned)
            chk_to_var = 2 * np.arctanh(chk_msg_prod[xx] / tanned)

            # var_to_chk
            marginal = priors + sum_cols(chk_to_var)
            var_to_chk = marginal[yy] - chk_to_var

            # bitwise decoding
            x_hat = (marginal < 0).astype(int)
            iter_count += 1


class MSA(SPA): pass
