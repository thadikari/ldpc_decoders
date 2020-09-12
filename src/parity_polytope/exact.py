from numpy.ctypeslib import ndpointer
from functools import wraps
import numpy as np
import ctypes
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(dir_path, 'ppolytope.lib')
global lib
lib = None

def init_lib():
    try:
        global lib
        lib = ctypes.cdll.LoadLibrary(lib_path)

        ndp_int = ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")
        ndp_dbl = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")

        lib.proj_vec.argtypes = [ctypes.c_size_t, ndp_dbl, ndp_dbl]
        lib.proj_csr.argtypes = [ctypes.c_size_t, ndp_int, ndp_dbl, ndp_dbl]

    except OSError as err:
        print('Caught OSError! Make sure a Python-32bit version is running.')
        print('https://stackoverflow.com/questions/19849077/error-loading-dll-in-python-not-a-valid-win32-application.')
        raise


# https://stackoverflow.com/questions/145270/calling-c-c-from-python
# https://stackoverflow.com/questions/5862915/passing-numpy-arrays-to-a-c-function-for-input-and-output
# https://stackoverflow.com/questions/5081875/ctypes-beginner


def require_init(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        if lib is None: init_lib()
        return func(*args, **kwargs)
    return func_wrapper

@require_init
def proj_vec(arr_in, arr_out=None):
    arr_out = np.zeros_like(arr_in, dtype=arr_in.dtype) if arr_out is None else arr_out
    lib.proj_vec(arr_in.size, arr_in, arr_out)
    return arr_out


@require_init
def proj_csr(csr_in):
    arr_out = np.zeros_like(csr_in.data, dtype=csr_in.data.dtype)
    lib.proj_csr(csr_in.indptr.size, csr_in.indptr, csr_in.data, arr_out)
    return arr_out


@require_init
def proj_rows(mat):
    arr_out = np.zeros(mat.size, dtype=mat.dtype)
    indptr = np.arange(mat.shape[0] + 1) * mat.shape[1]
    lib.proj_csr(indptr.size, indptr, mat[:], arr_out)
    return arr_out.reshape(mat.shape)


if __name__ == "__main__":
    import unittest


    class TestCase(unittest.TestCase):
        def test_vec(self):
            inp = np.array([1.5025, 0.5102, 1.0119, 1.3982, 1.7818, 1.9186, 1.0944, 0.2772, 0.2986, 0.5150])
            print(proj_vec(inp))

        def test_csr(self):
            from scipy.sparse import csr_matrix
            row = np.array([0, 0, 1, 2, 2, 2])
            col = np.array([0, 2, 2, 0, 1, 2])
            data = np.array([1., 0, 3, 1, 0, .6])
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
            csr = csr_matrix((data, (row, col)), shape=(3, 3), dtype=float)
            print(csr.toarray())
            print(proj_csr(csr))

        def test_mat(self):
            data = np.array([[0., .5, 1], [1, 1, 1], [1, 1, 0], [0, .4, .4]])
            print(proj_rows(data))


    unittest.main()
