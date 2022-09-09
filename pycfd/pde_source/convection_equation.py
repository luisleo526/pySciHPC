from typing import Callable

import numpy as np
from numba import int32, float64, njit

from pycfd.boundary_conditions import zero_order
from pycfd.functions.derivatives import find_fx, find_fy, find_fz


@njit(parallel=True, fastmath=True, nogil=True)
def pure_convection_source(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, vel: np.ndarray,
                           scheme: Callable):
    s = find_fx(f, grids[0], vel[0, :, :, :], scheme) * vel[0, :, :, :]
    if ndim > 1:
        s += find_fy(f, grids[1], vel[1, :, :, :], scheme) * vel[1, :, :, :]
    if ndim > 2:
        s += find_fz(f, grids[2], vel[2, :, :, :], scheme) * vel[2, :, :, :]
    return -s


@njit(fastmath=True, nogil=True)
def pure_convection(temproal: Callable, scheme: Callable, phi: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32,
                    velocity: np.ndarray, dt: float64):
    return temproal(dt, phi, grids, ghc, ndim, pure_convection_source, zero_order, velocity, scheme)
