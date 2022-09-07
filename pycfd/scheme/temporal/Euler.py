from typing import Callable

import numpy as np
from numba import njit, float64, int32


@njit(parallel=True, fastmath=True)
def euler(dt: float64, f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, find_source: Callable,
        boundary_condition: Callable, *args):
    s1 = find_source(f, grids, ghc, ndim, *args)
    f += s1 * dt
    boundary_condition(f, ghc, ndim)

    return f
