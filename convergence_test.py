from importlib import import_module
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, float64, int32

from boundary_conditions.periodic import periodic_1d_x
from methods.derivatives import find_fx
from objects.variable import Var
from scheme.temporal.runge_kutta import rk3
from utils.utils import find_order


@njit(parallel=True, fastmath=True)
def convection_source_1d(f: np.ndarray, dx: float64, ghc: int32, c: np.ndarray, scheme: Callable):
    return - find_fx(f, dx, c, scheme) * c


def run(N, source, bc, ghc, c, scheme, dt, plot=False):
    phi = Var([N + 1], ghc, [(-1.0, 1.0)])
    for i in range(phi.shape[0]):
        phi.data[i + phi.ghc, 0, 0] = np.sin(np.pi * phi.x[i] - np.sin(np.pi * phi.x[i]) / np.pi)
    bc(phi.data, phi.ghc)

    c_array = np.ones_like(phi.data) * c

    t = 0.0
    while t < 2.0:
        t += dt
        rk3(dt, phi.dx, phi.ghc, phi.data, source, bc, c_array, scheme)

    phi_exact = Var([N + 1], ghc, [(-1.0, 1.0)])
    for i in range(phi_exact.shape[0]):
        x = phi_exact.x[i] - c * (t - 2.0)
        phi_exact.data[i + phi_exact.ghc, 0, 0] = np.sin(np.pi * x - np.sin(np.pi * x) / np.pi)
    bc(phi_exact.data, phi_exact.ghc)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(phi.data[:, 0, 0])
        ax.plot(phi_exact.data[:, 0, 0])

    return np.linalg.norm(phi_exact.data[ghc:N + ghc + 1] - phi.data[ghc:N + ghc + 1]) / (N + 1)


if __name__ == "__main__":
    run_scheme = getattr(import_module("scheme.spatial"), input('Choose scheme (CCD, UCCD, WENO_JS, WENO_Z, CRWENO): '))
    data = {}
    for i in range(5, 10):
        data[2 ** i] = run(2 ** i, convection_source_1d, periodic_1d_x, 3, 1.0, run_scheme, 0.1 * 2.0 / 2 ** 9)
    print("---Positive speed---")
    find_order(data)
    data = {}
    for i in range(5, 10):
        data[2 ** i] = run(2 ** i, convection_source_1d, periodic_1d_x, 3, -1.0, run_scheme, 0.1 * 2.0 / 2 ** 9)
    print("---Negative speed---")
    find_order(data)
