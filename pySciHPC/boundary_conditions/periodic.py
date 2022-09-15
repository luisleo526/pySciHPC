import numpy as np
from numba import njit, prange, int32, float64

from pySciHPC.objects.base import Scalar


@njit((float64[:, :, :], int32), parallel=True, nogil=True)
def periodic_x(f: np.ndarray, ghc: int32):
    for j in prange(f.shape[1]):
        for k in prange(f.shape[2]):
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


@njit((float64[:, :, :], int32), parallel=True, nogil=True)
def periodic_z(f: np.ndarray, ghc: int32):
    for j in prange(f.shape[1]):
        for i in prange(f.shape[0]):
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

    # f.to_host()
    # periodic(f.data.cpu[0], geo.ghc, geo.ndim)
    # f.to_device()

    for j in range(f.data.gpu[0].shape[1]):
        for k in range(f.data.gpu[0].shape[2]):
            for i in range(geo.ghc):
                f.data.gpu[0, i, j, k] = f.data.gpu[0, -2 * geo.ghc - 1 + i, j, k]
                f.data.gpu[0, -i - 1, j, k] = f.data.gpu[0, 2 * geo.ghc - i, j, k]

    if geo.ndim > 1:
        for i in range(f.data.gpu[0].shape[0]):
            for k in range(f.data.gpu[0].shape[2]):
                for j in range(geo.ghc):
                    f.data.gpu[0, i, j, k] = f.data.gpu[0, i, -2 * geo.ghc - 1 + j, k]
                    f.data.gpu[0, i, -j - 1, k] = f.data.gpu[0, i, 2 * geo.ghc - j, k]

    if geo.ndim > 2:
        for j in range(f.data.gpu[0].shape[1]):
            for i in range(f.data.gpu[0].shape[0]):
                for k in range(geo.ghc):
                    f.data.gpu[0, i, j, k] = f.data.gpu[0, i, j, -2 * geo.ghc - 1 + k]
                    f.data.gpu[0, i, j, -k - 1] = f.data.gpu[0, i, j, 2 * geo.ghc - k]