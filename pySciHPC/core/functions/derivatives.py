from typing import Callable

import cupy as cp
import numpy as np
from numba import njit, float64, prange

from ..data import Scalar, Vector


@njit(parallel=True, fastmath=True, nogil=True)
def find_fx(f: np.ndarray, dx: float64, c: np.ndarray, scheme: Callable):
    fx = np.zeros_like(f)
    if f.shape[0] != 1:
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                fx[:, j, k] = scheme(f[:, j, k], c[:, j, k], dx)
    return fx


@njit(parallel=True, fastmath=True, nogil=True)
def find_fy(f: np.ndarray, dy: float64, c: np.ndarray, scheme: Callable):
    fy = np.zeros_like(f)
    if f.shape[1] != 1:
        for i in prange(f.shape[0]):
            for k in prange(f.shape[2]):
                fy[i, :, k] = scheme(f[i, :, k], c[i, :, k], dy)
    return fy


@njit(parallel=True, fastmath=True, nogil=True)
def find_fz(f: np.ndarray, dz: float64, c: np.ndarray, scheme: Callable):
    fz = np.zeros_like(f)
    if f.shape[2] != 1:
        for i in prange(f.shape[0]):
            for j in prange(f.shape[1]):
                fz[i, j, :] = scheme(f[i, j, :], c[i, j, :], dz)
    return fz


def cuda_find_fx(f: Scalar, geo: Scalar, vel: Vector, scheme: Callable, coeff, *args):
    cuda_fx = cp.zeros_like(f.data.gpu[0])
    if cuda_fx.shape[0] != 1:
        for j in range(cuda_fx.shape[1]):
            for k in range(cuda_fx.shape[2]):
                cuda_fx[:, j, k] = scheme(f.data.gpu[0, :, j, k], vel.x.data.gpu[0, :, j, k],
                                          geo.dx, f.blockspergrid[0], f.threadsperblock[0], coeff[0], *args)
    return cuda_fx


def cuda_find_fy(f: Scalar, geo: Scalar, vel: Vector, scheme: Callable, coeff, *args):
    cuda_fy = cp.zeros_like(f.data.gpu[0])
    if cuda_fy.shape[1] != 1:
        for i in range(cuda_fy.shape[0]):
            for k in range(cuda_fy.shape[2]):
                cuda_fy[i, : k] = scheme(f.data.gpu[0, i, :, k], vel.y.data.gpu[0, i, :, k],
                                         geo.dy, f.blockspergrid[1], f.threadsperblock[1], coeff[1], *args)
    return cuda_fy


def cuda_find_fz(f: Scalar, geo: Scalar, vel: Vector, scheme: Callable, coeff, *args):
    cuda_fz = cp.zeros_like(f.data.gpu[0])
    if cuda_fz.shape[2] != 1:
        for i in range(cuda_fz.shape[0]):
            for j in range(cuda_fz.shape[1]):
                cuda_fz[i, j, :] = scheme(f.data.gpu[0, i, j, :], vel.z.data.gpu[0, i, j, :],
                                          geo.dz, f.blockspergrid[2], f.threadsperblock[2], coeff[2], *args)
    return cuda_fz
