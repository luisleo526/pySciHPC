import numpy as np

from numba import njit, prange, float64, int32


@njit((float64[:, :, :], int32), parallel=True, nogil=True)
def zero_order_x(f: np.ndarray, ghc: int):
    for j in prange(f.shape[1]):
        for k in prange(f.shape[2]):
            for i in range(ghc):
                f[i, j, k] = f[ghc, j, k]
                f[-i - 1, j, k] = f[-ghc - 1, j, k]


@njit((float64[:, :, :], int32), parallel=True, nogil=True)
def zero_order_y(f: np.ndarray, ghc: int):
    for i in prange(f.shape[0]):
        for k in prange(f.shape[2]):
            for j in range(ghc):
                f[i, j, k] = f[i, ghc, k]
                f[i, -j - 1, k] = f[i, -ghc - 1, k]


@njit((float64[:, :, :], int32), parallel=True, nogil=True)
def zero_order_z(f: np.ndarray, ghc: int):
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in range(ghc):
                f[i, j, k] = f[i, j, ghc]
                f[i, j, -k - 1] = f[i, j, -ghc - 1]


@njit((float64[:, :, :], int32, int32), nogil=True)
def zero_order(f: np.ndarray, ghc: int, ndim: int):
    zero_order_x(f, ghc)
    if ndim > 1:
        zero_order_y(f, ghc)
    if ndim > 2:
        zero_order_z(f, ghc)


def zero_order_all(f: np.ndarray, ghc: int, ndim: int):
    for basis in range(f.shape[0]):
        zero_order(f[basis], ghc, ndim)
