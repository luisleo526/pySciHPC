from typing import Callable

import numpy as np
import cupy as cp
from numba import njit, float64, int32, cuda
from pySciHPC.objects.base import Scalar, Vector


@njit(parallel=True, fastmath=True, nogil=True)
def rk3(dt: float64, f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, find_source: Callable,
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


def cuda_rk3(dt, f: Scalar, geo: Scalar, vel: Vector, source: Callable, boundary_condition: Callable, *args):
    s1 = source(f, geo, vel, *args)
    f.data.gpu[0] += s1 * dt
    boundary_condition(f, geo)

    s2 = source(f, geo, vel, *args)
    f.data.gpu[0] += (-3.0 * s1 + s2) / 4.0 * dt
    boundary_condition(f, geo)

    s3 = source(f, geo, vel, *args)
    f.data.gpu[0] += (-s1 - s2 + 8.0 * s3) / 12.0 * dt
    boundary_condition(f, geo)
