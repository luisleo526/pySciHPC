import numpy as np
from numba import njit

from .basic import find_mass_vol, Heaviside, Delta
from .level_set_function import LevelSetFunction
from ..boundary_conditions.cell import zero_order
from ..functions.gradients import CCD_grad
from ..scheme.temporal.Runge_Kutta import rk3


@njit(parallel=True, fastmath=True, nogil=True)
def mpls_source(f: np.ndarray, grids: np.ndarray, ghc: int, ndim: int, delta0: np.ndarray, width: float, imass: float,
                density_ratio: float):
    dx, dy, dz = grids
    if ndim > 3:
        dv = dx * dy * dz
    else:
        dv = dx * dy
    vol, mass = find_mass_vol(f, width, density_ratio, dv, ndim, ghc)
    grad = CCD_grad(f, grids, ndim, ghc)
    h = Heaviside(f, width)
    delta = Delta(f, width)

    eta = (imass - mass) / np.sum(delta ** 2 * (2.0 * (1.0 - density_ratio) * h + density_ratio) * grad * dv)

    return eta * delta0 * grad


def solve_mpls(phi: LevelSetFunction, tol: float = 1e-10):
    while True:
        phi.data.cpu[0] = rk3(1.0, phi.data.cpu[0], phi.grids, phi.ghc, phi.ndim, mpls_source, zero_order, phi.delta,
                              phi.interface_width, phi.mass_history[0], phi.density_ratio)
        if abs(1.0 - phi.mass / phi.mass_history[0]) < tol:
            break
