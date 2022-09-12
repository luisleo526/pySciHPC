from typing import Callable

import numpy as np
import cupy as cp
from numba import int32, float64, njit

from pySciHPC.functions.derivatives import find_fx, find_fy, find_fz
from pySciHPC.functions.derivatives import cuda_find_fx, cuda_find_fy, cuda_find_fz
from pySciHPC.objects.base import Scalar, Vector


@njit(parallel=True, fastmath=True, nogil=True)
def pure_convection_source(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, vel: np.ndarray,
                           scheme: Callable, *args):
    s = find_fx(f, grids[0], vel[0, :, :, :], scheme, *args) * vel[0, :, :, :]
    if ndim > 1:
        s += find_fy(f, grids[1], vel[1, :, :, :], scheme, *args) * vel[1, :, :, :]
    if ndim > 2:
        s += find_fz(f, grids[2], vel[2, :, :, :], scheme, *args) * vel[2, :, :, :]
    return -s


def cuda_pure_convection_source(f: Scalar, geo: Scalar, vel: Vector, scheme: Callable, *args):
    s = - cuda_find_fx(f, geo, vel, scheme, *args) * vel.x.data.gpu[0]
    if geo.ndim > 1:
        s -= cuda_find_fy(f, geo, vel, scheme, *args) * vel.y.data.gpu[0]
    if geo.ndim > 2:
        s -= cuda_find_fz(f, geo, vel, scheme, *args) * vel.z.data.gpu[0]
    return s
