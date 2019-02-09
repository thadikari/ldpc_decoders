import numpy as np
import utils
import numpy.ma as ma


class Channel:
    def __init__(self, p):
        self.p = p

    def send(self, x):
        # any position = 2 means an erasure
        tt = (np.random.random(x.shape) < self.p).astype(int)
        return np.clip(x + tt * 10, 0, 2)


class ML:
    def __init__(self, p, code):
        self.log_p, self.log_1p = np.log(p), np.log(1 - p)
        self.cb = code.cb
        self.n = self.cb.shape[1]

    def decode(self, y):
        num_erasures = np.sum(y > 1)
        num_agrees = np.sum(self.cb == y, axis=1)
        num_diffs = self.n - num_agrees - num_erasures
        log_prob = num_erasures * self.log_p + num_agrees * self.log_1p
        log_prob[num_diffs > 0] = np.NINF  # CWs that don't match have NINF log likelihood
        ind = utils.arg_max_rand(log_prob)
        return self.cb[ind]


class SPA:
    def __init__(self, p, code):
        self.max_iter = 100
        self.symbols = np.array([-1, 1, 0])
        self.parity_mtx = ma.masked_array(code.parity_mtx.astype(int),
                                          mask=~code.parity_mtx.astype(bool))
        self.xx, self.yy = np.where(self.parity_mtx)
        self.xy = (self.xx, self.yy)

    def decode(self, y):
        xx, yy, xy = self.xx, self.yy, self.xy
        priors = self.symbols[y]
        chk_to_var = self.parity_mtx + 0
        var_to_chk = self.parity_mtx + 0
        var_to_chk[xy] = priors[yy]

        iter_count = 0
        x_hat = y

        while 1:
            iter_count += 1
            if iter_count > self.max_iter:
                # print('max reached', y.sum(), x_hat.sum())
                return x_hat

            # chk_to_var
            sums = (self.parity_mtx - np.abs(var_to_chk)).sum(axis=1)
            chk_to_var.data[sums > 1, :] = 0  # more than 1 erasures
            ind1 = sums == 1
            sums1 = np.abs(var_to_chk[ind1, :])
            sums2 = 2 * ((var_to_chk[ind1, :] > 0).astype(int).sum(axis=1) % 2) - 1
            chk_to_var.data[ind1, :] = (1 - sums1) * sums2[:, None]
            chk_to_var.data[sums == 0, :] = var_to_chk.data[sums == 0, :]

            # check if no erasures anymore
            marginal = np.sign(priors + chk_to_var.sum(axis=0))
            x_hat = np.array([2, 1, 0])[marginal]
            if np.sum(x_hat == 2) == 0: return x_hat

            # var_to_chk
            var_in_sum = chk_to_var.sum(axis=0) + priors
            var_to_chk.data[xy] = np.sign(var_in_sum[yy] - chk_to_var[xy])


class Test(utils.TestCase):
    def test_all(self):
        decoders = [ML, SPA]
        self.sample('4_2_test', 1 / 3, decoders,
                    [1, 1, 0, 1, 1],
                    [1, 2, 0, 1, 2])
        self.sample('7_4_hamming', .1, decoders,
                    [1, 0, 0, 1, 1, 0, 0],
                    [2, 0, 2, 1, 1, 0, 2])


if __name__ == "__main__":
    import unittest
    import codes

    np.random.seed(0)
    Test().test_all()  # unittest.main()
