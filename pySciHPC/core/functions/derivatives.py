from typing import Callable

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, nogil=True)
def find_fx_fxx(f: np.ndarray, dx: float, scheme: Callable):
    fx = np.zeros_like(f)
    fxx = np.zeros_like(f)
    if f.shape[0] != 1:
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                _fx, _fxx = scheme(f[:, j, k], dx)
                fx[:, j, k] = _fx
                fxx[:, j, k] = _fxx
    return np.stack((fx, fxx))


@njit(parallel=True, fastmath=True, nogil=True)
def find_fy_fyy(f: np.ndarray, dy: float, scheme: Callable):
    fy = np.zeros_like(f)
    fyy = np.zeros_like(f)
    if f.shape[1] != 1:
        for i in prange(f.shape[0]):
            for k in prange(f.shape[2]):
                _fy, _fyy = scheme(f[i, :, k], dy)
                fy[i, :, k] = _fy
                fyy[i, :, k] = _fyy
    return np.stack((fy, fyy))


@njit(parallel=True, fastmath=True, nogil=True)
def find_fz_fzz(f: np.ndarray, dz: float, scheme: Callable):
    fz = np.zeros_like(f)
    fzz = np.zeros_like(f)
    if f.shape[2] != 1:
        for i in prange(f.shape[0]):
            for j in prange(f.shape[1]):
                _fz, _fzz = scheme(f[i, j, :], dz)
                fz[i, j, :] = _fz
                fzz[i, j, :] = _fzz
    return np.stack((fz, fzz))
