import numpy as np


class Code:
    def __init__(self, gen_mtx, parity_mtx):
        self.gen_mtx, self.parity_mtx = gen_mtx, parity_mtx

        if gen_mtx is not None:
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
         '6_2_test': (None,
                      np.array([[1, 1, 1, 0, 0, 0],  # parity_mtx
                                [0, 0, 0, 1, 1, 1],
                                [0, 0, 1, 1, 0, 1],
                                [1, 1, 0, 0, 1, 0]])
                      ),
         '7_4_hamming': (np.array([[1, 1, 1, 0, 0, 0, 0],  # gen_mtx
                                   [1, 0, 0, 1, 1, 0, 0],
                                   [0, 1, 0, 1, 0, 1, 0],
                                   [1, 1, 0, 1, 0, 0, 1]]),
                         np.array([[0, 0, 0, 1, 1, 1, 1],  # parity_mtx
                                   [0, 1, 1, 0, 0, 1, 1],
                                   [1, 0, 1, 0, 1, 0, 1]])
                         ),
         '12_3_4_ldpc': (None,  # http://circuit.ucsd.edu/~yhk/ece154c-spr16/pdfs/ErrorCorrectionIII.pdf
                         np.array([[0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # parity_mtx
                                   [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                                   [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                                   [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                                   [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                                   [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
                                   [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]])
                         )
         }

file_codes = ['1200_3_6_ldpc']


def get_code_names():
    return list(codes.keys()) + file_codes


def get_code(name):
    if name in file_codes:
        return Code(None, load_parity_mtx(name))
    else:
        return Code(*codes[name])


def load_parity_mtx(name):
    mtx = np.zeros((600, 1200))
    with open('%s.txt' % name, 'r') as fp:
        chk_num = 1
        for line in fp:
            for var_num in map(int, line.split()):
                mtx[chk_num - 1, var_num - 1] = 1
            chk_num += 1
    return mtx
