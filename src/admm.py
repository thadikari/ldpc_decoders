from scipy.sparse import coo_matrix
import numpy as np

import math_utils as mu
from parity_polytope import exact
from parity_polytope import apprx


class ADMM_Base:
    id_keys = ['mu', 'eps', 'max_iter']

    def __init__(self, parity_mtx, **kwargs):
        self.mu, self.max_iter = kwargs['mu'], kwargs['max_iter']
        thresh = (kwargs['eps'] ** 2) * parity_mtx.sum()
        self.xx, self.yy = np.where(parity_mtx)
        self.var_deg = parity_mtx.sum(axis=0)
        self.is_close = lambda a_1, a_2: ((a_1 - a_2) ** 2).sum() < thresh
        self.coo = lambda d_: coo_matrix((d_, (self.xx, self.yy)), shape=parity_mtx.shape)
        self.sum_cols = lambda d_: mu.sum_axis(self.coo(d_), 0)

    def decode(self, y, gamma):
        xx, yy = self.xx, self.yy
        z_old, lambda_vec = yy * 0., yy * 0.
        x_hat, iter_count = y * 1., 0

        def ret(val):
            # print(val, ':', iter_count)
            return (x_hat > .5).astype(int)

        while 1:
            if 0 < self.max_iter <= iter_count: return ret('maximum')

            # update x
            x_hat[:] = np.clip((self.sum_cols(z_old - lambda_vec / self.mu)
                                - gamma / self.mu) / self.var_deg, 0., 1.)
            x_hat_yy = x_hat[yy]

            # update z
            v_vec = x_hat_yy + lambda_vec / self.mu
            z_new = self.project(v_vec)

            # update lambda
            lambda_vec[:] = lambda_vec + self.mu * (x_hat_yy - z_new)

            if self.is_close(x_hat_yy, z_new) and \
                    self.is_close(z_old, z_new): return ret('converged')
            z_old = z_new
            iter_count += 1


class ADMM(ADMM_Base):
    def __init__(self, parity_mtx, **kwargs):
        super().__init__(parity_mtx, **kwargs)

    def project(self, vec):
        return exact.proj_csr(self.coo(vec).tocsr())


class ADMMA(ADMM_Base):
    def __init__(self, parity_mtx, **kwargs):
        super().__init__(parity_mtx, **kwargs)
        dims = set(parity_mtx.sum(axis=1))
        if len(dims) != 1: raise Exception('Cannot use ADMMA decoder for codes with irregular check degree.')
        self.dim = dims.pop()
        self.model = apprx.load_model(dim=self.dim)

    def project(self, vec):
        ap = self.model.eval_rows(vec.reshape(-1, self.dim)).ravel()
        # ex = exact.proj_csr(self.coo(vec).tocsr())
        # print(abs(ap-ex).sum()/len(ex))
        return ap
