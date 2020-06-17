from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import numpy as np

import math_utils as mu
from parity_polytope import exact


class ADMM_Base:
    id_keys = ['mu', 'eps', 'max_iter', 'allow_pseudo']

    def __init__(self, parity_mtx, **kwargs):
        self.allow_pseudo = kwargs['allow_pseudo']
        self.mu, self.max_iter = kwargs['mu'], kwargs['max_iter']
        thresh = (kwargs['eps'] ** 2) * parity_mtx.sum()
        self.var_deg = parity_mtx.sum(axis=0)
        self.parity_mtx = parity_mtx

        def lambda_(x_hat_yy, z_old, z_new):
            aa1 = ((x_hat_yy - z_new) ** 2).sum()
            aa2 = ((z_old - z_new) ** 2).sum()
            # print('[', aa1, ',', aa2, '],')
            return aa1 < thresh and aa2 < thresh

        self.is_close = lambda_

        self.xx, self.yy = np.where(parity_mtx)
        dummy_data = self.yy * 0.
        sparse_ = lambda mat_: mat_((dummy_data, (self.xx, self.yy)), shape=parity_mtx.shape)
        self.coo_, self.csr_ = sparse_(coo_matrix), sparse_(csr_matrix)

        self.coo = lambda d_: mu.assign_data(self.coo_, d_)
        self.csr = lambda d_: mu.assign_data(self.csr_, d_)
        self.sum_cols = lambda d_: mu.sum_axis(self.coo(d_), 0)

        self.iter = np.zeros(2000, dtype=int)

    def stats(self):
        avg = self.iter @ np.arange(len(self.iter)) / self.iter.sum()
        return {'average': avg, 'iter': self.iter.tolist()}

    def decode(self, y, gamma):
        xx, yy = self.xx, self.yy
        z_old, lambda_vec, v_vec = yy * 0. + .5, yy * 0., yy * 1.
        x_hat, iter_count = y * 1., 0

        def ret(val):
            # print(val, ':', iter_count)
            self.iter[iter_count if iter_count < len(self.iter) else -1] += 1
            return mu.pseudo_to_cw(x_hat, self.allow_pseudo)

        while 1:
            if 0 < self.max_iter <= iter_count: return ret('maximum')

            # update x
            x_hat[:] = np.clip((self.sum_cols(z_old - lambda_vec / self.mu)
                                - gamma / self.mu) / self.var_deg, 0., 1.)
            x_hat_yy = x_hat[yy]

            # update z
            v_vec[:] = x_hat_yy + lambda_vec / self.mu
            z_new = self.projection(iter_count, v_vec)

            # update lambda
            lambda_vec[:] = lambda_vec + self.mu * (x_hat_yy - z_new)

            if self.is_close(x_hat_yy, z_old, z_new): return ret('converged')
            z_old = z_new
            iter_count += 1


class ADMM(ADMM_Base):
    def __init__(self, parity_mtx, **kwargs):
        super().__init__(parity_mtx, **kwargs)

    def projection(self, _, vec):
        return exact.proj_csr(self.csr(vec))


class ADMMA(ADMM_Base):
    id_keys = ADMM_Base.id_keys + ['layers']  # , 'apprx']

    def __init__(self, parity_mtx, **kwargs):
        from parity_polytope import apprx
        super().__init__(parity_mtx, **kwargs)
        dims = set(parity_mtx.sum(axis=1))
        if len(dims) != 1: raise Exception('Cannot use ADMMA decoder for codes with irregular check degree.')
        self.dim = dims.pop()
        self.train = 'train' in kwargs and kwargs['train']
        model_maker = apprx.make_model if self.train else apprx.load_model
        self.model = model_maker(dim=self.dim, layers=kwargs['layers'])
        if self.train: self.trainer = apprx.Trainer(self.model, save_freq=1000)
        self.switch = kwargs['apprx']

    def projection(self, iter_count, vec):
        if self.train:
            apx = exact.proj_csr(self.csr(vec))
            self.trainer.step(vec.reshape(-1, self.dim),
                              apx.reshape(-1, self.dim))
        else:
            if 0 < self.switch < iter_count:
                apx = exact.proj_csr(self.csr(vec))
            else:
                apx = self.model.eval_rows(vec.reshape(-1, self.dim)).ravel()
        # print(abs(ap-ex).sum()/len(ex))
        return apx
