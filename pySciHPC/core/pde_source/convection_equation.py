from typing import Callable

import numpy as np
from numba import njit

from ..functions.derivative import find_fx, find_fy, find_fz


@njit(parallel=True, fastmath=True, nogil=True)
def pure_convection_source(f: np.ndarray, grids: np.ndarray, ghc: int, ndim: int, vel: np.ndarray,
                           scheme: Callable):
    s = find_fx(f, grids[0], vel[0, :, :, :], scheme) * vel[0, :, :, :]
    if ndim > 1:
        s += find_fy(f, grids[1], vel[1, :, :, :], scheme) * vel[1, :, :, :]
    if ndim > 2:
        s += find_fz(f, grids[2], vel[2, :, :, :], scheme) * vel[2, :, :, :]
    return -s
