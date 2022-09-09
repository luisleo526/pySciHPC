from typing import Callable

import numpy as np
from numba import int32, float64, njit, prange

from pycfd.boundary_conditions import zero_order
from pycfd.functions.level_set import Heaviside, Delta
from pycfd.objects.level_set_function import find_mass_vol
from pycfd.utils import l2_norm


@njit(parallel=True, fastmath=True, nogil=True)
def mpls_source(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, c: np.ndarray, gradient: Callable,
                width: float64, imass: float64, density_ratio: float64):
    dx, dy, dz = grids
    if ndim > 3:
        dv = dx * dy * dz
    else:
        dv = dx * dy
    vol, mass = find_mass_vol(f, width, density_ratio, dv, ndim, ghc)
    grad = gradient(f, grids, ndim, ghc)
    h = Heaviside(f, width)
    delta = Delta(f, width)

    eta = (imass - mass) / np.sum(delta ** 2 * (2.0 * (1.0 - density_ratio) * h + density_ratio) * grad * dv)

    return eta * delta * grad


@njit(nogil=True, fastmath=True)
def mpls_criterion(f: np.ndarray, fold: np.ndarray, t: float64, tol: float64, period: float64, grids: np.ndarray,
                   ghc: int32, ndim: int32, gradient: Callable, width: float64, imass: float64,
                   density_ratio: float64):
    dx, dy, dz = grids
    if ndim > 3:
        dv = dx * dy * dz
    else:
        dv = dx * dy
    vol, mass = find_mass_vol(f, width, density_ratio, dv, ndim, ghc)
    return abs(1.0 - mass / imass) < tol
