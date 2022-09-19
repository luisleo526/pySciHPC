from typing import Callable

import numpy as np
from numba import njit, float64, prange

from ..scheme.spatial.CCD import CCD_full


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


@njit(parallel=True, fastmath=True, nogil=True)
def ccd_x(f: np.ndarray, dx: float64):
    fx = np.zeros_like(f)
    fxx = np.zeros_like(f)
    buffer = np.zeros(f.shape[0])
    if f.shape[0] != 1:
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                _fx, _fxx = CCD_full(f[:, j, k], buffer, dx)
                fx[:, j, k] = _fx
                fxx[:, j, k] = _fxx
    return np.stack((fx, fxx))


@njit(parallel=True, fastmath=True, nogil=True)
def ccd_y(f: np.ndarray, dy: float64):
    fy = np.zeros_like(f)
    fyy = np.zeros_like(f)
    buffer = np.zeros(f.shape[1])
    if f.shape[1] != 1:
        for i in prange(f.shape[0]):
            for k in prange(f.shape[2]):
                _fy, _fyy = CCD_full(f[i, :, k], buffer, dy)
                fy[i, :, k] = _fy
                fyy[i, :, k] = _fyy
    return np.stack((fy, fyy))


@njit(parallel=True, fastmath=True, nogil=True)
def ccd_z(f: np.ndarray, dz: float64):
    fz = np.zeros_like(f)
    fzz = np.zeros_like(f)
    buffer = np.zeros(f.shape[2])
    if f.shape[2] != 1:
        for i in prange(f.shape[0]):
            for j in prange(f.shape[1]):
                _fz, _fzz = CCD_full(f[i, j, :], buffer, dz)
                fz[i, j, :] = _fz
                fzz[i, j, :] = _fzz
    return np.stack((fz, fzz))
