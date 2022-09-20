import numpy as np
from numba import float64, prange, njit


@njit(float64[:](float64[:], float64[:], float64), parallel=True, fastmath=True, nogil=True)
def QUICK(f: np.ndarray, c: np.ndarray, dx: float):
    fh = np.zeros_like(f)
    fx = np.zeros_like(f)
    for i in prange(1, fh.size - 2):
        if c[i] + c[i + 1] >= 0.0:
            fh[i] = (-f[i - 1] + 6.0 * f[i] + 3.0 * f[i + 1]) / 8.0
        else:
            fh[i] = (-f[i + 2] + 6.0 * f[i + 1] + 3.0 * f[i]) / 8.0
    for i in prange(1, fh.size):
        fx[i] = (fh[i] - fh[i - 1]) / dx
    return fx
