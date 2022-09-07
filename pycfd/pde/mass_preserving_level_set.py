from typing import Callable

import numpy as np
from numba import int32, float64, njit, prange

from pycfd.boundary_conditions import zero_order
from pycfd.functions.level_set import Heaviside, Delta
from pycfd.objects.level_set_function import find_mass_vol


@njit(parallel=True, fastmath=True)
def mpls_source(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, gradient: Callable, width: float64,
                imass: float64, density_ratio: float64):
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


@njit(fastmath=True, parallel=True)
def mpls(temproal: Callable, gradient: Callable, phi: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32,
         width: float64, imass: float64, density_ratio: float64):
    dx, dy, dz = grids
    if ndim > 3:
        dv = dx * dy * dz
    else:
        dv = dx * dy

    while True:
        phi = temproal(1.0, phi, grids, ghc, ndim, mpls_source, zero_order, gradient, width, imass, density_ratio)
        vol, mass = find_mass_vol(phi, width, density_ratio, dv, ndim, ghc)

        if abs(1.0 - mass / imass) < 1.0e-10:
            break

    return phi
