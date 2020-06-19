import numpy as np
import argparse
import os

import math_utils as mu


class Code:
    def __init__(self, gen_mtx, parity_mtx):
        self.gen_mtx, self.parity_mtx = gen_mtx, parity_mtx

        if gen_mtx is not None:
            k, n = gen_mtx.shape
            messages = mu.binary_vectors(k)
            self.cb = (messages @ gen_mtx) % 2

            # check if GH^T = 0
            assert (np.sum((self.cb @ parity_mtx.T) % 2) == 0)
            assert (self.cb[0].sum() == 0)  # all zeros cw
            # assert (self.cb[-1].sum() == n)  # all ones cw

    def get_k(self): return self.get_n() - self.parity_mtx.shape[0]

    def get_n(self): return self.parity_mtx.shape[1]


codes = {'4_2_test': (np.array([[1, 1, 1, 0, 0],  # gen_mtx
                                [0, 0, 1, 1, 1]]),
                      np.array([[1, 1, 0, 0, 0],  # parity_mtx
                                [0, 1, 1, 1, 0],
                                [0, 0, 0, 1, 1]])
                      ),
         # find gen_mtx using wolframalpha: nullspace of <H> in GF(2)
         '6_2_3_ldpc': (np.array([[0, 0, 0, 1, 0, 1],  # gen_mtx
                                  [1, 0, 1, 1, 1, 0],
                                  [1, 1, 0, 0, 0, 0]]),
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
         # http://circuit.ucsd.edu/~yhk/ece154c-spr16/pdfs/ErrorCorrectionIII.pdf
         '12_3_4_ldpc': (np.array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                                   [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                                   [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                                   [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1]]),
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

file_codes_dir_string = 'FILE_CODES_DIR'
file_codes_dir = os.environ.get(file_codes_dir_string, os.path.join('data', 'codes'))
file_codes_dir = os.path.abspath(file_codes_dir)


def get_file_code_map():
    file_list = next(os.walk(file_codes_dir), ((), (), ()))[2]
    file_code_list = list(map(lambda x: os.path.splitext(x)[0], file_list))
    file_code_map = dict(zip(file_code_list, file_list))
    return file_code_map


def get_code_names():
    return list(codes.keys()) + list(get_file_code_map().keys())


def get_code(name):
    file_code_map = get_file_code_map()
    if name in get_file_code_map().keys():
        file_path = os.path.join(file_codes_dir, file_code_map[name])
        return Code(None, load_parity_mtx(file_path))
    else:
        return Code(*codes[name])


def load_parity_mtx(file_path):
    with open(file_path, 'r') as fp:
        lines = tuple(line for line in fp if len(line.split()) > 0)
        max_ind = max(tuple(max(map(int, line.split())) for line in lines))
        min_ind = min(tuple(min(map(int, line.split())) for line in lines))
        if min_ind not in [0, 1]: raise Exception('Minimum index is not 0 or 1.')
        mtx = np.zeros((len(lines), max_ind + (0 if min_ind == 1 else 1)), int)
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


def save_parity_mtx(parity_mtx, code_name):
    file_path = os.path.join(file_codes_dir, '%s.txt' % code_name)
    with open(file_path, 'w') as fp:
        for chk_ind in range(parity_mtx.shape[0]):
            ind = np.where(parity_mtx[chk_ind, :])[0] + 1
            fp.writelines(' '.join(map(str, ind)) + '\n')


def gen_rand_reg_ldpc(args):
    n, l, r = args.n, args.l, args.r
    for i in range(args.count):
        parity_mtx = rand_reg_ldpc(n, l, r)
        code_name = '%d_%d_%d_rand_ldpc_%d' % (n, l, r, i + 1)
        save_parity_mtx(parity_mtx, code_name)
        verify_rand_reg_ldpc(code_name, l, r)


def verify_rand_reg_ldpc(code_name, l, r):
    parity_mtx = get_code(code_name).parity_mtx
    print(parity_mtx.shape,
          (parity_mtx.sum(axis=0) == l).all(),
          (parity_mtx.sum(axis=1) == r).all())


# find_gen_mtx given parity_mtx, not final version.
def find_gen_mtx():
    H = 0  # copy the parity_mtx of 12_3_4_ldpc code
    all_sets = mu.binary_vectors(12).T
    G = all_sets[:, (H @ all_sets % 2).sum(0) == 0][:, [1, 2, 4, 8, 16]].T
    print(G @ H.T % 2, G.shape)
    print(G)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('count', help='number of random codes to generate', type=int)
    parser.add_argument('n', help='regular ldpc code length', type=int)
    parser.add_argument('l', help='l', type=int)
    parser.add_argument('r', help='r', type=int)
    return parser


if __name__ == "__main__":
    gen_rand_reg_ldpc(setup_parser().parse_args())
