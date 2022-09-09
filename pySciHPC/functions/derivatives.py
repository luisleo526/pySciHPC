from typing import Callable

import numpy as np
from numba import njit, float64, prange


@njit(parallel=True, fastmath=True, nogil=True)
def find_fx(f: np.ndarray, dx: float64, c: np.ndarray, scheme: Callable, *args):
    fx = np.zeros_like(f)
    if f.shape[0] != 1:
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                fx[:, j, k] = scheme(f[:, j, k], c[:, j, k], dx, *args)
    return fx


@njit(parallel=True, fastmath=True, nogil=True)
def find_fy(f: np.ndarray, dy: float64, c: np.ndarray, scheme: Callable, *args):
    fy = np.zeros_like(f)
    if f.shape[1] != 1:
        for i in prange(f.shape[0]):
            for k in prange(f.shape[2]):
                fy[i, :, k] = scheme(f[i, :, k], c[i, :, k], dy, *args)
    return fy


@njit(parallel=True, fastmath=True, nogil=True)
def find_fz(f: np.ndarray, dz: float64, c: np.ndarray, scheme: Callable, *args):
    fz = np.zeros_like(f)
    if f.shape[2] != 1:
        for i in prange(f.shape[0]):
            for j in prange(f.shape[1]):
                fz[i, j, :] = scheme(f[i, j, :], c[i, j, :], dz, *args)
    return fz
