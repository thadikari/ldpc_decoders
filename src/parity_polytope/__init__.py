from numpy.ctypeslib import ndpointer
import numpy as np
import ctypes
import os

lib_path = os.path.join(os.environ['SCRATCH'], 'ppolytope.so')
lib = ctypes.cdll.LoadLibrary(lib_path)
fun_proj = lib.proj_2d_arr
fun_proj.restype, fun_proj.argtypes = None, [ctypes.c_size_t,
                                             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]


def proj_pp(arr_in, arr_out=None):
    arr_out = np.zeros_like(arr_in, dtype=arr_in.dtype) if arr_out is None else arr_out
    fun_proj(arr_in.size, arr_in, arr_out)
    return arr_out


def test_all():
    inp = np.array([1.5025, 0.5102, 1.0119, 1.3982, 1.7818, 1.9186, 1.0944, 0.2772, 0.2986, 0.5150])
    print(inp)
    print(proj_pp(inp))
    # testlib.myprint()


if __name__ == "__main__":
    test_all()
