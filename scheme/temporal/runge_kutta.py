from typing import Callable

import numpy as np
from numba import njit, float64, int32


@njit(parallel=True, fastmath=True)
def rk3(dt: float64, dx: float64, ghc: int32, f: np.ndarray, find_source: Callable, boundary_condition: Callable,
        *args):
    s1 = find_source(f, dx, ghc, *args)
    f += s1 * dt
    boundary_condition(f, ghc)

    s2 = find_source(f, dx, ghc, *args)
    f += (-3.0 * s1 + s2) / 4.0 * dt
    boundary_condition(f, ghc)

    s3 = find_source(f, dx, ghc, *args)
    f += (-s1 - s2 + 8.0 * s3) / 12.0 * dt
    boundary_condition(f, ghc)
