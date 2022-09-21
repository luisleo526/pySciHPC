import numpy as np
from numba import njit, float64, prange


@njit(float64[:](float64[:], float64[:], float64), parallel=True, fastmath=True, nogil=True)
def UpwindSecondOrder(f: np.ndarray, c: np.ndarray, dx: float):
    fx = np.zeros_like(f)
    for i in prange(2, f.size - 2):
        if c[i] > 0.0:
            fx[i] = 0.5 * (f[i - 2] - 4.0 * f[i - 1] + 3.0 * f[i]) / dx
        else:
            fx[i] = -0.5 * (f[i + 2] - 4.0 * f[i + 1] + 3.0 * f[i]) / dx
    return fx


@njit(float64[:](float64[:], float64[:], float64), parallel=True, fastmath=True, nogil=True)
def CentralSecondOrder(f: np.ndarray, c: np.ndarray, dx: float):
    fx = np.zeros_like(f)
    for i in prange(1, f.size - 1):
        fx[i] = 0.5 * (f[i + 1] - f[i - 1]) / dx
    return fx


@njit(float64[:](float64[:], float64[:], float64), parallel=True, fastmath=True, nogil=True)
def CentralSecondOrder_full(f: np.ndarray, dx: float):
    fx = np.zeros_like(f)
    fxx = np.zeros_like(f)
    for i in prange(1, f.size - 1):
        fx[i] = 0.5 * (f[i + 1] - f[i - 1]) / dx
        fxx[i] = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / dx ** 2.0
    return np.stack((fx, fxx))
