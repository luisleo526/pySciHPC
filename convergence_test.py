from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, float64, prange

from boundary_conditions.periodic import periodic_1d
from methods.find_fx import find_fx
from objects.variable import create_Var, Var
from scheme.temporal.runge_kutta import rk3
from utils.utils import find_order
from importlib import import_module


@njit(parallel=True, fastmath=True)
def convection_source_1d(f: Var, c: np.ndarray, scheme: Callable):
    return - find_fx(f, c, scheme) * c


def run(N, source, bc, ghc, c, scheme, dt, plot=False):
    phi = create_Var([N + 1], ghc, [(-1.0, 1.0)])
    for i in range(phi.shape[0]):
        phi.data[i + phi.ghc, 0, 0] = np.sin(np.pi * phi.x_axis[i])
    periodic_1d(phi)

    c_array = np.ones_like(phi.data) * c

    t = 0.0
    while t < 2.0:
        t += dt
        rk3(dt, phi, source, bc, c_array, scheme)

    phi_exact = create_Var([N + 1], ghc, [(-1.0, 1.0)])
    for i in range(phi_exact.shape[0]):
        phi_exact.data[i + phi_exact.ghc, 0, 0] = np.sin(np.pi * (phi_exact.x_axis[i] - c * (t - 2.0)))
    periodic_1d(phi_exact)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(phi.data[:, 0, 0])
        ax.plot(phi_exact.data[:, 0, 0])

    return np.linalg.norm(phi_exact.data[ghc:N + ghc + 1] - phi.data[ghc:N + ghc + 1]) / (N + 1)


if __name__ == "__main__":
    run_scheme = getattr(import_module("scheme.spatial"), input('Choose scheme (CCD, UCCD, WENO_JS, WENO_Z, CRWENO): '))
    data = {}
    for i in range(5, 10):
        data[2 ** i] = run(2 ** i, convection_source_1d, periodic_1d, 3, 1.0, run_scheme, 0.01 * 2.0 / 2 ** 9)
    find_order(data)
    data = {}
    for i in range(5, 10):
        data[2 ** i] = run(2 ** i, convection_source_1d, periodic_1d, 3, -1.0, run_scheme, 0.01 * 2.0 / 2 ** 9)
    find_order(data)
