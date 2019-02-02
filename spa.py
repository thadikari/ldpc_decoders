import numpy as np
import utils


class SPA:
    def __init__(self, parity_mtx):
        self.max_iter = 100
        self.parity_mtx = parity_mtx

        [c, n] = self.parity_mtx.shape

        self.var_neigh = tuple(np.transpose(np.nonzero(self.parity_mtx[:, ind])).flatten() for ind in range(n))
        self.chk_neigh = tuple(np.transpose(np.nonzero(self.parity_mtx[ind, :])).flatten() for ind in range(c))

        self.chk_combs = []
        for chk_ind in range(c):
            degree = self.chk_neigh[chk_ind].shape[0]
            power = 2 ** np.arange(degree)
            d = np.arange(2 ** degree)
            permutations = np.floor((d[:, None] % (2 * power)) / power).astype(int)
            combs = permutations[permutations.sum(axis=1) % 2 == 0]
            num_combs = combs.shape[0]

            # number of terms in marginalization for var = 0 OR 1
            # half of terms denote var = 0 and other half var = 1
            num_terms = num_combs >> 1  # divide by 2
            # which combinations correspond to var_i = 0, i in 0th dim
            combs_whr_0 = np.where(combs.T == 0)[1].reshape([-1, num_terms])
            # which combinations correspond to var_i = 1, i in 0th dim
            combs_whr_1 = np.where(combs.T == 1)[1].reshape([-1, num_terms])
            self.chk_combs.append((combs, combs_whr_0, combs_whr_1))

    def decode(self, y, priors):

        [c, n] = self.parity_mtx.shape

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
                degree = var_indices.shape[0]

                chk_msg_in = var_to_chk[chk_ind, var_indices]
                chk_in_exp = np.exp(chk_msg_in)
                log_prob_0 = chk_in_exp / (1 + chk_in_exp)
                log_prob_1 = 1 / (1 + chk_in_exp)
                aug_chk_msg_in = np.vstack((log_prob_0, log_prob_1)).T

                # pre computed valid var combinations
                combs, combs_whr_0, combs_whr_1 = self.chk_combs[chk_ind]
                # each row --> probs according to each combination
                # includes the prob of receiving variable as well
                # need to be subtracted next
                prob_combs = aug_chk_msg_in[np.arange(degree), combs]
                prod = prob_combs.prod(axis=1)  # multiply probs

                # removing the contribution from the receiving variable
                # sum_terms_0: terms to be summed in numerator
                sum_terms_0 = prod[combs_whr_0] / log_prob_0[:, None]
                sum_terms_1 = prod[combs_whr_1] / log_prob_1[:, None]

                chk_msg_out = np.log(sum_terms_0.sum(axis=1)) - np.log(sum_terms_1.sum(axis=1))
                chk_to_var[chk_ind, var_indices] = chk_msg_out

            # check if a codeword
            x_hat = ((priors + chk_to_var.sum(axis=0)) < 0) + 0
            if not ((self.parity_mtx @ x_hat) % 2).any():
                # if ~(y == x_hat).all(): print('111')
                return x_hat

            # var_to_chk
            for var_ind in range(n):
                chk_indices = self.var_neigh[var_ind]
                var_msg_in = chk_to_var[chk_indices, var_ind]
                prod = var_msg_in.sum() + priors[var_ind]
                var_msg_out = prod - var_msg_in
                var_to_chk[chk_indices, var_ind] = var_msg_out
