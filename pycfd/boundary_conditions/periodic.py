import numpy as np
from numba import njit, prange, int32, float64


@njit((float64[:, :, :], int32), parallel=True)
def periodic_x(f: np.ndarray, ghc: int32):
    for j in prange(f.shape[1]):
        for k in prange(f.shape[2]):
            for i in range(ghc):
                f[i, j, k] = f[f.shape[0] - 2 * ghc + i - 1, j, k]
                f[f.shape[0] - ghc + i, j, k] = f[ghc + i + 1, j, k]


@njit((float64[:, :, :], int32), parallel=True)
def periodic_y(f: np.ndarray, ghc: int32):
    for i in prange(f.shape[0]):
        for k in prange(f.shape[2]):
            for j in range(ghc):
                f[i, j, k] = f[i, f.shape[1] - 2 * ghc + j - 1, k]
                f[i, f.shape[1] - ghc + j, k] = f[i, ghc + j + 1, k]


@njit((float64[:, :, :], int32), parallel=True)
def periodic_z(f: np.ndarray, ghc: int32):
    for j in prange(f.shape[1]):
        for i in prange(f.shape[0]):
            for k in range(ghc):
                f[i, j, k] = f[i, j, f.shape[2] - 2 * ghc + k - 1]
                f[i, j, f.shape[2] - ghc + k] = f[i, j, ghc + k + 1]


@njit((float64[:, :, :], int32, int32))
def periodic(f: np.ndarray, ghc: int32, ndim: int32):
    periodic_x(f, ghc)
    if ndim > 1:
        periodic_y(f, ghc)
    if ndim > 2:
        periodic_z(f, ghc)
