from scipy.sparse import coo_matrix
import math_utils as mu
import numpy as np


class BPA:
    id_keys = ['max_iter']

    def __init__(self, parity_mtx, **kwargs):
        self.max_iter = kwargs['max_iter']
        self.parity_mtx = parity_mtx
        self.xx, self.yy = np.where(self.parity_mtx)

        self.coo = lambda d_: coo_matrix((d_, (self.xx, self.yy)), shape=parity_mtx.shape)
        self.sum_cols = lambda d_: mu.sum_axis(self.coo(d_), 0)

    def decode(self, y, priors):
        xx, yy, sum_cols = self.xx, self.yy, self.sum_cols
        var_to_chk, chk_to_var = priors[yy], priors[yy]
        x_hat, iter_count = y, 0

        def ret(val):
            # print(val, ':', iter_count)
            return x_hat

        # np.seterr(all='raise')
        while 1:
            if 0 < self.max_iter <= iter_count: return ret('maximum')
            if ((self.parity_mtx @ x_hat) % 2 == 0).all(): return ret('decoded')

            # chk_to_var
            self.decode_(var_to_chk, xx, chk_to_var)

            # var_to_chk
            marginal = priors + sum_cols(chk_to_var)
            if 1:
                var_to_chk = marginal[yy] - chk_to_var
                marginal[np.isnan(marginal)] = 0.
            else:
                # TODO: following piece is to handle cases where llr is inf. slower and uglier than inf unhandled case. needs refactoring
                # assert fails on (inf-inf), i.e., conflicting 100% beliefs form checks
                # may be replace them by 0 cuz conflicting certainties mean uncertain?
                # assert (~np.isnan(marginal).any())
                marginal[np.isnan(marginal)] = 0.
                mar_yy = marginal[yy]

                inf_ind = np.abs(chk_to_var) == np.inf
                sums = sum_cols(inf_ind)
                ma_0, ma_1, ma_2 = (sums == 0)[yy], (sums == 1)[yy], (sums >= 2)[yy]
                # cols with 0 inf elements
                var_to_chk[ma_0] = mar_yy[ma_0] - chk_to_var[ma_0]
                # cols with 1 or more +-inf elements get +-inf
                var_to_chk[~ma_0] = mar_yy[~ma_0] - 0.

                # cols with 1 +-inf element
                chk_to_var[inf_ind] = 0.
                chk_to_var_inf = (priors + sum_cols(chk_to_var))[yy]
                ma_1 = np.logical_and(ma_1, inf_ind, out=ma_1)
                var_to_chk[ma_1] = chk_to_var_inf[ma_1]

            # bitwise decoding
            x_hat = (marginal < 0).astype(int)
            iter_count += 1


class SPA(BPA):
    def __init__(self, parity_mtx, **kwargs):
        super().__init__(parity_mtx, **kwargs)
        self.prod_rows = lambda d_: mu.prod_nonzero(self.coo(d_), 1)

    def decode_(self, var_to_chk, xx, chk_to_var):
        tanned = np.tanh(var_to_chk / 2.)
        chk_msg_prod = self.prod_rows(tanned)
        tan = chk_msg_prod[xx] / tanned  # TODO: handle possible div by 0
        chk_to_var[:] = 2 * mu.arctanh(tan, out=chk_to_var)


class MSA(BPA):
    def __init__(self, parity_mtx, **kwargs):
        super().__init__(parity_mtx, **kwargs)
        self.sign_rows = lambda d_: mu.prod_nonzero_sign(self.coo(d_), 1)
        self.to_csr = lambda d_: self.coo(d_).tocsr()
        self.min_rows = lambda csr: mu.csr_csc_argmax(-csr)
        self.row_ind = np.array(range(parity_mtx.shape[0]))

    def decode_(self, var_to_chk, xx, chk_to_var):
        # TODO: way slower than SPA. needs refactoring
        sign = self.sign_rows(var_to_chk)[xx] / mu.sign(var_to_chk)
        csr = self.to_csr(np.abs(var_to_chk))
        chk_to_var_csr = self.to_csr(chk_to_var)
        arg_min_1 = self.min_rows(csr)
        min_ind_1 = self.row_ind, arg_min_1
        min_val_1 = mu.mtx_to_vec(csr[min_ind_1])
        chk_to_var_csr.data[:] = min_val_1[xx]

        csr[min_ind_1] = np.inf
        arg_min_2 = self.min_rows(csr)
        min_ind_2 = self.row_ind, arg_min_2
        min_val_2 = mu.mtx_to_vec(csr[min_ind_2])
        chk_to_var_csr[min_ind_1] = min_val_2

        chk_to_var[:] = sign * chk_to_var_csr.data
