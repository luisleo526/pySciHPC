from typing import Callable

import numpy as np

from .convection_equation import pure_convection_source
from ..functions.cell_face_interpolation import all_face_x
from ..functions.stress_tensor import CCD_stress


def single_phase_flows(vel: np.ndarray, grids: np.ndarray, ghc: int, ndim: int, vel0: np.ndarray, scheme: Callable,
                       viscosity: np.ndarray, params):
    stress = CCD_stress(vel, grids, ndim, viscosity)

    x_vel = all_face_x(vel0, ndim)
    sx = pure_convection_source(vel[0], grids, ghc, ndim, x_vel, scheme) + stress[0] / params.Re
