import numpy as np
import os


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

file_codes_dir = os.path.join('..', 'codes')
file_list = next(os.walk(file_codes_dir))[2]
file_code_list = list(map(lambda x: os.path.splitext(x)[0], file_list))
file_code_map = dict(zip(file_code_list, file_list))


def get_code_names():
    return list(codes.keys()) + file_code_list


def get_code(name):
    if name in file_code_list:
        file_path = os.path.join(file_codes_dir, file_code_map[name])
        return Code(None, load_parity_mtx(file_path))
    else:
        return Code(*codes[name])


def load_parity_mtx(file_path):
    with open(file_path, 'r') as fp:
        lines = tuple(line for line in fp if len(line.split()) > 0)
        max_ind = max(tuple(max(map(int, line.split())) for line in lines))
        mtx = np.zeros((len(lines), max_ind), int)
        chk_num = 1
        for line in lines:
            for var_num in map(int, line.split()):
                mtx[chk_num - 1, var_num - 1] = 1
            chk_num += 1
    return mtx


def rand_reg_ldpc(n, l, r):
    m = int(n * l / r)
    parity_mtx = np.zeros((m, n), int)
    var_ind = np.arange(n)
    for i in range(m):
        pairs = list(zip(parity_mtx.sum(axis=0), var_ind))
        np.random.shuffle(pairs)
        pairs.sort(key=lambda x: x[0])
        ind = np.array(list(zip(*pairs))[1])
        parity_mtx[i, ind[0:r]] = 1
    assert ((parity_mtx.sum(axis=0) == l).all())
    assert ((parity_mtx.sum(axis=1) == r).all())
    return parity_mtx  # np.sort(H @ (2 ** var_ind))


def rand_reg_ldpc_test():
    nn = 100
    xx = np.zeros([nn, 6])
    for i in range(nn):
        xx[i, :] = rand_reg_ldpc(12, 3, 6)
    print(xx[xx[:, 0].argsort()])


def gen_rand_reg_ldpc():
    n, l, r = 512, 3, 6
    for i in range(1):
        parity_mtx = rand_reg_ldpc(n, l, r)
        file_path = os.path.join(file_codes_dir,
                                 '%d_%d_%d_rand_ldpc_%d.txt'
                                 % (n, l, r, i + 1))
        with open(file_path, 'w') as fp:
            for chk_ind in range(parity_mtx.shape[0]):
                ind = np.where(parity_mtx[chk_ind, :])[0] + 1
                fp.writelines(' '.join(map(str, ind)) + '\n')


def verify_rand_reg_ldpc():
    parity_mtx = get_code('512_3_6_ldpc_1').parity_mtx
    print(parity_mtx.shape,
          (parity_mtx.sum(axis=0) == 3).all(),
          (parity_mtx.sum(axis=1) == 6).all())


if __name__ == "__main__":
    # gen_rand_reg_ldpc()
    verify_rand_reg_ldpc()
