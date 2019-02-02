import numpy as np


class Code:
    def __init__(self, gen_mtx, parity_mtx):
        self.gen_mtx, self.parity_mtx = gen_mtx, parity_mtx

        k, n = gen_mtx.shape
        d = np.arange(2 ** k)
        messages = ((d[:, None] & (1 << np.arange(k))) > 0).astype(int)
        self.cb = (messages @ gen_mtx) % 2

        # check if GH^T = 0
        assert (np.sum((self.cb @ parity_mtx.T) % 2) == 0)
        assert (self.cb[0].sum() == 0)  # all zeros cw
        # assert (self.cb[-1].sum() == n)  # all ones cw
        # print(cb)


codes = {'4_2_test': (np.array([[1, 1, 1, 0, 0],  # gen_mtx
                                [0, 0, 1, 1, 1]]),
                      np.array([[1, 1, 0, 0, 0],  # parity_mtx
                                [0, 1, 1, 1, 0],
                                [0, 0, 0, 1, 1]])
                      ),
         '7_4_hamming': (np.array([[1, 1, 1, 0, 0, 0, 0],  # gen_mtx
                                   [1, 0, 0, 1, 1, 0, 0],
                                   [0, 1, 0, 1, 0, 1, 0],
                                   [1, 1, 0, 1, 0, 0, 1]]),
                         np.array([[0, 0, 0, 1, 1, 1, 1],  # parity_mtx
                                   [0, 1, 1, 0, 0, 1, 1],
                                   [1, 0, 1, 0, 1, 0, 1]])
                         )
         }


def get_code_names():
    return codes.keys()


def get_code(name):
    return Code(*codes[name])
