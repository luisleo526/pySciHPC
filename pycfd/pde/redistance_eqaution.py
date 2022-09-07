from typing import Callable

from numba import int32, boolean, float64, njit, prange

from pycfd.boundary_conditions import zero_order
from pycfd.functions.gradients import Godunov_WENO_grad
from pycfd.functions.level_set import Delta, Sign
from pycfd.utils import l2_norm

import numpy as np


@njit(float64[:, :, :](float64[:, :, :], float64[:], int32, int32, float64[:, :, :], float64, boolean), parallel=True,
      nogil=True)
def redistance_source(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, sign: np.ndarray, ls_width: float64,
                      init: bool):
    _lambda = np.zeros_like(f)
    grad = Godunov_WENO_grad(f, grids, ghc, ndim)
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


@njit(parallel=True, fastmath=True, nogil=True)
def solve_redistance(temproal: Callable, phi: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, ls_width: float64,
                     dt: float64, period: float64, init: bool):
    if init:
        phi = phi / np.amax(Godunov_WENO_grad(phi, grids, ghc, ndim))

    sign0 = Sign(phi, ls_width)

    t = 0
    cnt = 0
    while True:
        cnt += 1
        t += dt
        phi_tmp = np.copy(phi)
        phi = temproal(dt, phi, grids, ghc, ndim, redistance_source, zero_order, sign0, ls_width, init)
        error = l2_norm(phi_tmp, phi)
        if cnt % 100 == 0:
            print(cnt, error)
        if error < 1.0e-14 or t > period:
            break

    return phi
