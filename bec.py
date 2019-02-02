import numpy as np
import utils


class Channel:
    def __init__(self, p):
        self.p = p

    def send(self, x):
        # any negative position means an erasure
        tt = (np.random.random(x.shape) < self.p).astype(int)
        return np.clip(x + tt * -10, -3, 1)


class ML:
    def __init__(self, p, code):
        self.log_p, self.log_1p = np.log(p), np.log(1 - p)
        self.cb = code.cb
        self.n = self.cb.shape[1]

    def decode(self, y):
        num_erasures = np.sum(y < 0)
        num_agrees = np.sum(self.cb == y, axis=1)
        num_diffs = self.n - num_agrees - num_erasures
        log_prob = num_erasures * self.log_p + num_agrees * self.log_1p
        log_prob[num_diffs > 0] = np.NINF  # CWs that don't match have NINF log likelihood
        ind = utils.arg_max_rand(log_prob)
        return self.cb[ind]


class SPA:
    def __init__(self, p, code):
        self.max_iter = 100
        self.cb = code.parity_mtx

    def decode(self, y):
        iter_count = 0
        x_hat = y * 0
        return x_hat
        # while 1:
        #     iter_count += 1
        #     if iter_count > self.max_iter: return x_hat
        #
        #     check_ele_matrix = x_est(check_pos_matrix);
        #     erasures_matrix = check_ele_matrix > 2;
        #     erasure_counts = sum(erasures_matrix, 2);
        #     correctable_check_rows = find(erasure_counts == 1);
        #
        #     if isempty(correctable_check_rows)
        #         if sum(erasure_counts)
        #             logTxt('can not fully decode!');
        #         else
        #             logTxt('fully decoded!');
        #         end
        #         break
        #     end
        #
        #     check_rows = check_pos_matrix(correctable_check_rows', :);
        #     check_rows_erasures = erasures_matrix(correctable_check_rows', :);
        #     check_rows_non_erasures = ~check_rows_erasures;
        #     check_value_indexes = check_rows(check_rows_non_erasures);
        #     temp_length = length(correctable_check_rows);
        #     check_values_size = [temp_length, length(check_value_indexes)/temp_length];
        #     check_values = reshape(check_value_indexes, check_values_size);
        #     beliefs = mod(sum(x_est(check_values), 2), 2);
        #     x_est(check_rows(check_rows_erasures)) = beliefs;
        # end


class Test(utils.TestCase):
    def test_all(self):
        decoders = [ML, SPA]
        self.sample('4_2_test', 1 / 3, decoders,
                    [1, 1, 0, 1, 1],
                    [1, -3, 0, 1, -3])
        self.sample('7_4_hamming', .1, decoders,
                    [1, 0, 0, 1, 1, 0, 0],
                    [-3, -3, -3, 1, 1, 0, 0])


if __name__ == "__main__":
    import unittest
    import codes

    np.random.seed(0)
    Test().test_all()  # unittest.main()
