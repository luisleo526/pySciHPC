from typing import Callable

import cupy as cp
import numpy as np
from numba import int32, njit

from pySciHPC.cuda_solvers.derivatives_solver import CudaDerivativesSolver
from pySciHPC.functions.derivatives import find_fx, find_fy, find_fz
from pySciHPC.objects.base import Scalar, Vector
from .kernel_functions import neg_multi_sum_init, neg_multi_sum


@njit(parallel=True, fastmath=True, nogil=True)
def pure_convection_source(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, vel: np.ndarray,
                           scheme: Callable):
    s = find_fx(f, grids[0], vel[0, :, :, :], scheme) * vel[0, :, :, :]
    if ndim > 1:
        s += find_fy(f, grids[1], vel[1, :, :, :], scheme) * vel[1, :, :, :]
    if ndim > 2:
        s += find_fz(f, grids[2], vel[2, :, :, :], scheme) * vel[2, :, :, :]
    return -s


def cuda_pure_convection_source(f: Scalar, geo: Scalar, vel: Vector, solver: CudaDerivativesSolver, s: cp.ndarray,
                                *args):
    neg_multi_sum_init(solver.find_fx(f.data.gpu[0], vel.x.data.gpu[0]), vel.x.data.gpu[0], s)
    if f.ndim > 1:
        neg_multi_sum(solver.find_fy(f.data.gpu[0], vel.y.data.gpu[0]), *vel.y.data.gpu[0], s)
    if f.ndim > 2:
        neg_multi_sum(solver.find_fz(f.data.gpu[0], vel.z.data.gpu[0]), *vel.z.data.gpu[0], s)
