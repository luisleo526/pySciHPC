import time
from importlib import import_module
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, float64, int32

from pySciHPC.boundary_conditions import periodic
from pySciHPC.functions.derivatives import find_fx
from pySciHPC.objects import Scalar
from pySciHPC.scheme.temporal import rk3
from pySciHPC.utils import find_order, l2_norm
from pySciHPC import solve_hyperbolic
from pySciHPC.pde_source.convection_equation import pure_convection_source


@njit(parallel=True, fastmath=True, nogil=True)
def BurgerEq_source_1d(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, c: np.ndarray, scheme: Callable):
    return - find_fx(0.5 * f ** 2, grids[0], c, scheme)


@njit(float64[:](float64[:], float64, float64), parallel=True, fastmath=True, nogil=True)
def find_exact_solution(x: np.ndarray, t: float64, ts: float64):
    u = np.sin(2.0 * np.pi * x) / (2.0 * np.pi * ts)
    diff = u - np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)
    while np.linalg.norm(diff) / diff.size > 1.0e-14:
        u = np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)
        diff = u - np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)

    return u


def run(N, source, bc, ghc, ts, scheme, dt, plot=False):
    geo_dict = dict(_size=[N], ghc=ghc, _axis_data=[(0.0, 1.0)])

    geo = Scalar(**geo_dict, no_data=True)
    phi = Scalar(**geo_dict, no_axis=True)

    phi.core = np.sin(2.0 * np.pi * geo.mesh.x) / (2.0 * np.pi * ts)
    bc(phi.data.cpu[0], phi.ghc, phi.ndim)

    cpu_time = -time.time()
    t = 0.0
    while t < 0.75 * ts:
        t += dt
        solve_hyperbolic(phi, phi.data.cpu[0], geo, rk3, bc, source, dt, scheme)

    phi_exact = Scalar(**geo_dict, no_axis=True)
    phi_exact.core = find_exact_solution(geo.mesh.x, t, ts)
    bc(phi_exact.data.cpu[0], phi_exact.ghc, phi_exact.ndim)

    error = l2_norm(phi_exact.core, phi.core)
    return error, cpu_time + time.time()


if __name__ == "__main__":
    run_scheme = getattr(import_module("pySciHPC.scheme.spatial"),
                         input('Choose scheme (CCD, UCCD, WENO_JS, WENO_Z, CRWENO, CRWENO_LD): '))
    data = {}
    for i in range(5, 11):
        data[2 ** i] = run(2 ** i, BurgerEq_source_1d, periodic, 3, 2.0, run_scheme, 0.1 * 1.0 / 2 ** 10)
    find_order(data)
