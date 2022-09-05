import numpy as np

from numba import njit, prange, float64, int32


@njit((float64[:, :, :], int32), parallel=True)
def zero_order_x(f: np.ndarray, ghc: int32):
    for j in prange(f.shape[1]):
        for k in prange(f.shape[2]):
            for i in range(ghc):
                f[i, j, k] = f[ghc, j, k]
                f[f.shape[0] - ghc + i, j, k] = f[f.shape[0] + ghc - 1, j, k]


@njit((float64[:, :, :], int32), parallel=True)
def zero_order_y(f: np.ndarray, ghc: int32):
    for i in prange(f.shape[0]):
        for k in prange(f.shape[2]):
            for j in range(ghc):
                f[i, j, k] = f[i, ghc, k]
                f[i, f.shape[1] - ghc + j, k] = f[i, f.shape[1] + ghc - 1, k]


@njit((float64[:, :, :], int32), parallel=True)
def zero_order_z(f: np.ndarray, ghc: int32):
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in range(ghc):
                f[i, j, k] = f.data[i, j, ghc]
                f[i, j, f.shape[2] - ghc + k] = f[i, j, f.shape[2] + ghc - 1]