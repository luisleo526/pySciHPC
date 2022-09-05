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
def BurgerEq_source_1d(f: np.ndarray, dx: float64, ghc: int32, c: np.ndarray, scheme: Callable):
    return - find_fx(0.5 * f ** 2, dx, c, scheme)


@njit(float64[:](float64[:], float64, float64), parallel=True, fastmath=True)
def find_exact_solution(x: np.ndarray, t: float64, ts: float64):
    u = np.sin(2.0 * np.pi * x) / (2.0 * np.pi * ts)
    diff = u - np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)
    while np.linalg.norm(diff) / diff.size > 1.0e-14:
        u = np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)
        diff = u - np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)

    return u


def run(N, source, bc, ghc, ts, scheme, dt, plot=False):
    phi = Var([N + 1], ghc, [(0.0, 1.0)])
    for i in range(phi.shape[0]):
        phi.data[i + phi.ghc, 0, 0] = np.sin(2.0 * np.pi * phi.x[i]) / (2.0 * np.pi * ts)
    bc(phi.data, phi.ghc)

    t = 0.0
    while t < 0.75 * ts:
        t += dt
        rk3(dt, phi.dx, phi.ghc, phi.data, source, bc, phi.data, scheme)

    phi_exact = Var([N + 1], ghc, [(0.0, 1.0)])
    phi_exact.data[ghc:N + ghc + 1, 0, 0] = find_exact_solution(phi_exact.x, t, ts)
    bc(phi_exact.data, phi_exact.ghc)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(phi.data[:, 0, 0])
        ax.plot(phi_exact.data[:, 0, 0])

    return np.linalg.norm(phi_exact.data[ghc:N + ghc + 1] - phi.data[ghc:N + ghc + 1]) / (N + 1)


if __name__ == "__main__":
    run_scheme = getattr(import_module("scheme.spatial"), input('Choose scheme (CCD, UCCD, WENO_JS, WENO_Z, CRWENO): '))
    data = {}
    for i in range(5, 9):
        data[2 ** i] = run(2 ** i, BurgerEq_source_1d, periodic_1d_x, 3, 2.0, run_scheme, 0.1 * 1.0 / 2 ** 8)
    find_order(data)
