from typing import Callable

import numpy as np
from numba import njit, float64

from .convection_equation import pure_convection_source
from ..functions.cell_face_interpolation import all_face_x, all_face_z, all_face_y
from ..functions.stress_tensor import ccd_stress


@njit(float64[:, :, :, :], parallel=True, fastmath=True, nogil=True)
def single_phase_flows(vel: np.ndarray, grids: np.ndarray, ghc: int, ndim: int, vel0: np.ndarray, scheme: Callable,
                       viscosity: np.ndarray, params):
    stress = ccd_stress(vel, grids, ndim, viscosity)

    x_vel = all_face_x(vel0, ndim)
    sx = pure_convection_source(vel[0], grids, ghc, ndim, x_vel, scheme) + stress[0] / params.Re

    y_vel = all_face_y(vel0, ndim)
    sy = pure_convection_source(vel[1], grids, ghc, ndim, y_vel, scheme) + stress[1] / params.Re

    if ndim > 2:
        z_vel = all_face_z(vel0, ndim)
        sz = pure_convection_source(vel[2], grids, ghc, ndim, z_vel, scheme) + stress[2] / params.Re
        return np.stack((sx, sy, sz))
    else:
        return np.stack((sx, sy))
