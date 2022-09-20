import numpy as np
from numba import njit, float64, prange, int32
from pySciHPC.utils.utils import l2_norm


@njit(float64[:, :, :](float64[:, :, :], float64), fastmath=True, parallel=True, nogil=True)
def Heaviside(x, eta):
    heaviside = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                if x[i, j, k] > eta:
                    heaviside[i, j, k] = 1.0
                elif x[i, j, k] < -eta:
                    heaviside[i, j, k] = 0.0
                else:
                    heaviside[i, j, k] = 0.5 * (1.0 + x[i, j, k] / eta + np.sin(np.pi * x[i, j, k] / eta) / np.pi)
    return heaviside


@njit(float64[:, :, :](float64[:, :, :], float64), fastmath=True, parallel=True, nogil=True)
def Sign(x, eta):
    sign = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                if x[i, j, k] > 0.0:
                    sign[i, j, k] = 1.0
                else:
                    sign[i, j, k] = -1.0
                if abs(x[i, j, k]) < 1.0e-14:
                    sign[i, j, k] = 0.0

    return sign


@njit(float64[:, :, :](float64[:, :, :], float64), fastmath=True, parallel=True, nogil=True)
def Delta(x, eta):
    delta = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                if abs(x[i, j, k]) > eta:
                    delta[i, j, k] = 0.0
                else:
                    delta[i, j, k] = 0.5 * (1.0 + np.cos(np.pi * x[i, j, k] / eta)) / eta
    return delta


@njit(float64[:, :, :](float64[:, :, :], float64, float64), parallel=True, fastmath=True, nogil=True)
def smooth_with_heaviside(phi: np.ndarray, width: float, coef: float):
    h = Heaviside(phi, width)
    return h + (1.0 - h) * coef


@njit(float64[:](float64[:, :, :], float64, float64, float64, int32, int32), fastmath=True, parallel=True, nogil=True)
def find_mass_vol(x, eta, density, dv, ndim, ghc):
    h = Heaviside(x, eta)
    vol = 0.0
    mass = 0.0
    if ndim == 2:
        for i in prange(ghc, x.shape[0] - ghc):
            for j in prange(ghc, x.shape[1] - ghc):
                for k in prange(x.shape[2]):
                    vol += h[i, j, k] * dv
                    mass += h[i, j, k] * (h[i, j, k] + (1.0 - h[i, j, k]) * density) * dv
    else:
        for i in prange(ghc, x.shape[0] - ghc):
            for j in prange(ghc, x.shape[1] - ghc):
                for k in prange(ghc, x.shape[2] - ghc):
                    vol += h[i, j, k] * dv
                    mass += h[i, j, k] * (h[i, j, k] + (1.0 - h[i, j, k]) * density) * dv
    return np.array([vol, mass], dtype='float64')


@njit(float64(float64[:, :, :], float64[:, :, :], float64), fastmath=True, nogil=True)
def inteface_error(f: np.ndarray, g: np.ndarray, interface_width: float):
    h1 = Heaviside(f, interface_width)
    h2 = Heaviside(g, interface_width)
    return l2_norm(h1, h2)
