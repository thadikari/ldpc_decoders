import numpy as np
import utils


class SPA:
    def __init__(self, parity_mtx):
        self.max_iter = 100
        self.parity_mtx = parity_mtx

        [c, n] = self.parity_mtx.shape
        self.xx, self.yy = np.where(self.parity_mtx)
        self.chk_neigh = tuple(np.transpose(np.nonzero(self.parity_mtx[ind, :])).flatten() for ind in range(c))

    def decode(self, y, priors):

        [c, n] = self.parity_mtx.shape
        xx, yy = self.xx, self.yy

        var_to_chk = self.parity_mtx @ np.diag(priors)
        chk_to_var = self.parity_mtx * 0.

        iter_count = 0
        x_hat = y

        while 1:
            iter_count += 1
            if iter_count > self.max_iter: return x_hat

            # chk_to_var
            for chk_ind in range(c):
                # indices of connected variables
                var_indices = self.chk_neigh[chk_ind]
                chk_msg_in = var_to_chk[chk_ind, var_indices]

                if 0:  # min sum
                    tiled = np.tile(chk_msg_in, (chk_msg_in.shape[0], 1))
                    np.fill_diagonal(tiled, 1e5)
                    sign = np.sign(tiled).prod(axis=1)
                    np.fill_diagonal(tiled, np.Inf)
                    min = np.min(np.abs(tiled), axis=1)
                    chk_msg_out = sign * min
                else:  # sum prod
                    tiled = np.tile(chk_msg_in / 2., (chk_msg_in.shape[0], 1))
                    tanned = np.tanh(tiled)
                    np.fill_diagonal(tanned, 1.)
                    prod = tanned.prod(axis=1)
                    chk_msg_out = 2 * np.arctanh(prod)

                chk_to_var[chk_ind, var_indices] = chk_msg_out

            # check if a codeword
            x_hat = ((priors + chk_to_var.sum(axis=0)) < 0) + 0
            if not ((self.parity_mtx @ x_hat) % 2).any():
                # if ~(y == x_hat).all(): print('111')
                return x_hat

            # var_to_chk
            var_in_sum = chk_to_var.sum(axis=0) + priors
            var_to_chk[xx, yy] = var_in_sum[yy] - chk_to_var[xx, yy]
