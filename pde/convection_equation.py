from typing import Callable

import numpy as np
from numba import int32, float64, njit

from boundary_conditions.zero_order import zero_order
from functions.derivatives import find_fx, find_fy, find_fz


@njit(float64[:, :, :], parallel=True, fastmath=True)
def convection_source(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, vel: np.ndarray, scheme: Callable):
    s = find_fx(f, grids[0], vel[0, :, :, :], scheme) * vel[0, :, :, :] + \
        find_fy(f, grids[1], vel[1, :, :, :], scheme) * vel[1, :, :, :]
    if ndim > 2:
        s += find_fz(f, grids[2], vel[2, :, :, :], scheme) * vel[2, :, :, :]
    return -s


@njit(float64[:, :, :], fastmath=True)
def solve_convection(temproal: Callable, scheme: Callable, phi: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32,
                     velocity: np.ndarray, dt: float64):
    return temproal(dt, phi, grids, ghc, ndim, convection_source, zero_order, velocity, scheme)
