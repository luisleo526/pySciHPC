from typing import Callable

import numpy as np
from numba import njit, float64, int32

from pySciHPC.cuda_solvers.derivatives_solver import CudaDerivativesSolver
from pySciHPC.objects.base import Scalar, Vector
from .kernel_functions import rk3_1, rk3_2, rk3_3


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


def cuda_rk3(dt, f: Scalar, geo: Scalar, vel: Vector, source: Callable, boundary_condition: Callable,
             solver: CudaDerivativesSolver, *args):
    source(f, geo, vel, solver, solver.src_buffer0, *args)
    rk3_1(solver.src_buffer0, solver.dt, f.data.gpu[0])
    boundary_condition(f, geo)

    source(f, geo, vel, solver, solver.src_buffer1, *args)
    rk3_2(solver.src_buffer0, solver.src_buffer1, solver.dt, f.data.gpu[0])
    boundary_condition(f, geo)

    source(f, geo, vel, solver, solver.src_buffer2, *args)
    rk3_3(solver.src_buffer0, solver.src_buffer1, solver.src_buffer2, solver.dt, f.data.gpu[0])
    boundary_condition(f, geo)
