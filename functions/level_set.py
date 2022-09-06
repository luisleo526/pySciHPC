import numpy as np
from numba import njit, float64, prange


@njit(float64[:, :, :](float64[:, :, :], float64), fastmath=True, parallel=True)
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


@njit(float64[:, :, :](float64[:, :, :], float64), fastmath=True, parallel=True)
def Sign(x, eta):
    return Heaviside(x, eta) * 2.0 - 1.0


@njit(float64[:, :, :](float64[:, :, :], float64), fastmath=True, parallel=True)
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
