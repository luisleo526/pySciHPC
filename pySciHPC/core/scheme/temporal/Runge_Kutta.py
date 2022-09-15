from typing import Callable

import numpy as np
from numba import njit


@njit(parallel=True, fastmath=True, nogil=True)
def rk3(dt: float, f: np.ndarray, grids: np.ndarray, ghc: int, ndim: int, find_source: Callable,
        boundary_condition: Callable, c: np.ndarray, *args):
    s1 = find_source(f, grids, ghc, ndim, c, *args)
    f += s1 * dt
    boundary_condition(f, ghc, ndim)

    s2 = find_source(f, grids, ghc, ndim, c, *args)
    f += (-3.0 * s1 + s2) / 4.0 * dt
    boundary_condition(f, ghc, ndim)

    s3 = find_source(f, grids, ghc, ndim, c, *args)
    f += (-s1 - s2 + 8.0 * s3) / 12.0 * dt
    boundary_condition(f, ghc, ndim)

    return f
