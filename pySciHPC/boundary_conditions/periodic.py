import numpy as np
import cupy as cp
from numba import njit, prange, int32, float64, cuda

from pySciHPC.objects.base import Scalar


@njit((float64[:, :, :], int32), parallel=True, nogil=True)
def periodic_x(f: np.ndarray, ghc: int32):
    for j in prange(f.shape[1]):
        for k in prange(f.shape[2]):
            for i in range(ghc):
                f[i, j, k] = f[-2 * ghc - 1 + i, j, k]
                f[-i - 1, j, k] = f[2 * ghc - i, j, k]


@cuda.jit
def cuda_periodic_x(f, ghc_array):
    j, k = cuda.grid(2)
    if j < f.shape[1] and k < f.shape[2]:
        ghc = ghc_array[0, j, k]
        for i in range(ghc):
            f[i, j, k] = f[-2 * ghc - 1 + i, j, k]
            f[-i - 1, j, k] = f[2 * ghc - i, j, k]


@njit((float64[:, :, :], int32), parallel=True, nogil=True)
def periodic_y(f: np.ndarray, ghc: int32):
    for i in prange(f.shape[0]):
        for k in prange(f.shape[2]):
            for j in range(ghc):
                f[i, j, k] = f[i, -2 * ghc - 1 + j, k]
                f[i, -j - 1, k] = f[i, 2 * ghc - j, k]


@cuda.jit
def cuda_periodic_y(f, ghc_array):
    i, k = cuda.grid(2)
    if i < f.shape[0] and k < f.shape[2]:
        ghc = ghc_array[i, 0, k]
        for j in range(ghc):
            f[i, j, k] = f[i, -2 * ghc - 1 + j, k]
            f[i, -j - 1, k] = f[i, 2 * ghc - j, k]


@njit((float64[:, :, :], int32), parallel=True, nogil=True)
def periodic_z(f: np.ndarray, ghc: int32):
    for j in prange(f.shape[1]):
        for i in prange(f.shape[0]):
            for k in range(ghc):
                f[i, j, k] = f[i, j, -2 * ghc - 1 + k]
                f[i, j, -k - 1] = f[i, j, 2 * ghc - k]


@cuda.jit
def cuda_periodic_z(f, ghc_array):
    i, j = cuda.grid(2)
    if i < f.shape[0] and j < f.shape[1]:
        ghc = ghc_array[i, j, 0]
        for k in range(ghc):
            f[i, j, k] = f[i, j, -2 * ghc - 1 + k]
            f[i, j, -k - 1] = f[i, j, 2 * ghc - k]


@njit((float64[:, :, :], int32, int32))
def periodic(f: np.ndarray, ghc: int32, ndim: int32):
    periodic_x(f, ghc)
    if ndim > 1:
        periodic_y(f, ghc)
    if ndim > 2:
        periodic_z(f, ghc)


def cuda_periodic(f: Scalar, geo: Scalar):
    # ghc_array = cp.ones_like(f.data.gpu[0], dtype='int32') * geo.ghc
    # cuda_periodic_x[geo.blockspergrid_jk, geo.threadsperblock_jk](f.data.gpu[0], ghc_array)
    # if geo.ndim > 1:
    #     cuda_periodic_y[geo.blockspergrid_ik, geo.threadsperblock_ik](f.data.gpu[0], ghc_array)
    # if geo.ndim > 2:
    #     cuda_periodic_z[geo.blockspergrid_ij, geo.threadsperblock_ij](f.data.gpu[0], ghc_array)
    # del ghc_array

    f.to_host()
    periodic(f.data.cpu[0], geo.ghc, geo.ndim)
    f.to_device()
