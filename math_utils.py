import numpy as np

mtx_to_vec = lambda mtx: np.asarray(mtx).flatten()

sum_axis = lambda coo, axis: mtx_to_vec(coo.sum(axis=axis))


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
