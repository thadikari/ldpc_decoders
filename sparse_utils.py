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
