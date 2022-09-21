import numpy as np
from numba import njit, float64, int32, boolean, prange

from .basic import Delta, Sign
from .level_set_function import LevelSetFunction
from ..boundary_conditions.cell import zero_order
from ..functions.gradients import godunov_wenojs
from ..scheme.temporal import rk3


@njit(float64[:, :, :](float64[:, :, :], float64[:], int32, int32), fastmath=True, parallel=True, nogil=True)
def stabilize(f: np.ndarray, grids: np.ndarray, ghc: int, ndim: int):
    f = f / np.amax(godunov_wenojs(f, grids, ghc, ndim))
    return f


@njit(float64[:, :, :](float64[:, :, :], float64[:], int32, int32, float64[:, :, :], float64, boolean),
      parallel=True, nogil=True, fastmath=True)
def redistance_source(f: np.ndarray, grids: np.ndarray, ghc: int, ndim: int, sign: np.ndarray, ls_width: float,
                      init: bool):
    _lambda = np.zeros_like(f)
    grad = godunov_wenojs(f, grids, ghc, ndim)
    delta = Delta(f, ls_width)

    if not init:

        num = sign * delta * (grad - 1.0)
        den = grad * delta ** 2

        zero_order(num, ghc, ndim)
        zero_order(den, ghc, ndim)

        if ndim == 2:
            k = 0
            for i in prange(1, f.shape[0] - 1):
                for j in prange(1, f.shape[1] - 1):
                    int_num = 15.0 * num[i, j, k] + np.sum(num[i - 1:i + 2, j - 1:j + 2, k])
                    int_den = 15.0 * den[i, j, k] + np.sum(den[i - 1:i + 2, j - 1:j + 2, k])
                    if abs(int_den) > 1.0e-14:
                        _lambda[i, j, k] = int_num / int_den
        else:
            for i in prange(1, f.shape[0] - 1):
                for j in prange(1, f.shape[1] - 1):
                    for k in prange(1, f.shape[2] - 1):
                        int_num = 50.0 * num[i, j, k] + np.sum(num[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                        int_den = 50.0 * den[i, j, k] + np.sum(den[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                        if abs(int_den) > 1.0e-14:
                            _lambda[i, j, k] = int_num / int_den

        zero_order(_lambda, ghc, ndim)

    return - sign * (grad - 1.0) + _lambda * delta * grad


def solve_redistance(phi: LevelSetFunction, period: float, cfl: float, init: bool):
    sign0 = Sign(phi.data.cpu[0], phi.interface_width)
    if init:
        phi.data.cpu[0] = stabilize(phi.data.cpu[0], phi.grids, phi.ghc, phi.ndim)

    dt = cfl * phi.h
    t = 0.0

    while t < period:
        t += dt
        tmp = np.copy(phi.data.cpu[0])
        phi.data.cpu[0] = rk3(dt, phi.data.cpu[0], phi.grids, phi.ghc, phi.ndim, redistance_source, zero_order, sign0,
                              phi.interface_width, init)
        if np.amax(phi.data.cpu[0] - tmp) < 1.0e-10:
            break
