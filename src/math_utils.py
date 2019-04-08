import scipy.sparse as sp
import numpy as np
import itertools

mtx_to_vec = lambda mtx: np.asarray(mtx).ravel()

sum_axis = lambda coo, axis: mtx_to_vec(coo.sum(axis=axis))

# np.sign like func with no zeros returned
sign = lambda val: (val >= 0).astype(int) * 2 - 1


def assign_data(mat_, d_):
    mat_.data = d_
    return mat_


# all binary_vectors of given length in a matrix
def binary_vectors(length):
    if 0:
        d = np.arange(2 ** length)
        return ((d[:, None] & (1 << np.arange(length))) > 0).astype(int)
    else:
        str_seq = [seq for seq in itertools.product("01", repeat=length)]
        return np.array(str_seq).astype(np.int)


def pseudo_to_cw(x_, allow_pseudo, eps=1e-8):
    if allow_pseudo:
        x_[x_ < eps] = 0
        x_[1 - x_ < eps] = 1
        return x_
    else:
        return (x_ > .5).astype(int)


# input should be a coo_sparse mtx
def prod_nonzero_sign(coo, axis):
    temp = coo.data
    coo.data = coo.data < 0
    out = (mtx_to_vec(coo.sum(axis=axis)) % 2) * -2 + 1
    coo.data = temp
    return out


# input should be a coo_sparse mtx, axis= 0 cols, 1 rows
def prod_nonzero(coo, axis):
    temp = coo.data  # reusing existing coo without creating new one
    coo.data = np.log(np.abs(coo.data))
    mag = np.exp(mtx_to_vec(coo.sum(axis=axis)))
    coo.data = temp
    return prod_nonzero_sign(coo, axis) * mag


# wrapper to avoid the warning from np.arctanh(1 or -1)
def arctanh(val, out):
    ind_inf = np.abs(val) == 1
    out[ind_inf] = np.inf * val[ind_inf]
    out[~ind_inf] = np.arctanh(val[~ind_inf])
    return out


def log_sum_exp_rows(arr):
    arr_max = arr.max(axis=1)
    return arr_max + np.log(np.exp(arr - arr_max[:, None]).sum(axis=1))
    # sum_terms_1 = np.array([[1, 2, 3], [6, -1, -6]])
    # print(sum_terms_1)
    # print(log_sum_exp_rows(sum_terms_1))


# returns a random element if encountered multiple arg maxes
def arg_max_rand(values):
    max_ind = np.argwhere(values == np.max(values))
    return np.random.choice(max_ind.flatten(), 1)[0]


# https://stackoverflow.com/questions/30742572/argmax-of-each-row-or-column-in-scipy-sparse-matrix
def csr_csc_argmax(X, axis=None):
    is_csr = isinstance(X, sp.csr_matrix)
    is_csc = isinstance(X, sp.csc_matrix)
    assert (is_csr or is_csc)
    assert (not axis or (is_csr and axis == 1) or (is_csc and axis == 0))

    major_size = X.shape[0 if is_csr else 1]
    major_lengths = np.diff(X.indptr)  # group_lengths
    major_not_empty = (major_lengths > 0)

    result = -np.ones(shape=(major_size,), dtype=X.indices.dtype)
    split_at = X.indptr[:-1][major_not_empty]
    maxima = np.zeros((major_size,), dtype=X.dtype)
    maxima[major_not_empty] = np.maximum.reduceat(X.data, split_at)
    all_argmax = np.flatnonzero(np.repeat(maxima, major_lengths) == X.data)
    result[major_not_empty] = X.indices[all_argmax[np.searchsorted(all_argmax, split_at)]]
    return result
