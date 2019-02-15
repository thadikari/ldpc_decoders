import numpy as np


class SPA:
    def __init__(self, parity_mtx, max_iter):
        self.max_iter = max_iter
        self.parity_mtx = parity_mtx
        self.xx, self.yy = np.where(self.parity_mtx)
        self.xy = (self.xx, self.yy)

    def decode(self, y, priors):

        xx, yy, xy = self.xx, self.yy, self.xy
        var_to_chk = self.parity_mtx @ np.diag(priors)
        chk_to_var = self.parity_mtx * 0.

        iter_count = 0
        x_hat = y

        while 1:
            iter_count += 1
            if iter_count > self.max_iter:
                print('max reached', y.sum(), x_hat.sum())
                return x_hat

            # chk_to_var
            chk_msg = np.ones_like(self.parity_mtx, float)
            tanned = np.tanh(var_to_chk[xy] / 2.)
            chk_msg[xy] = tanned
            chk_msg_prod = chk_msg.prod(axis=1)
            chk_to_var[xy] = 2 * np.arctanh(chk_msg_prod[xx] / tanned)

            # check if a codeword
            x_hat = ((priors + chk_to_var.sum(axis=0)) < 0).astype(int)
            if not ((self.parity_mtx @ x_hat) % 2).any(): return x_hat

            # var_to_chk
            var_in_sum = chk_to_var.sum(axis=0) + priors
            var_to_chk[xy] = var_in_sum[yy] - chk_to_var[xy]


class MSA(SPA): pass
