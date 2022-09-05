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


@njit(parallel=True, fastmath=True, cache=True)
def BurgerEq_source_1d(f: Var, c: np.ndarray, scheme: Callable):
    return - find_fx(0.5 * f.data ** 2, f.dx, c, scheme)


@njit(float64[:](float64[:], float64, float64), parallel=True, fastmath=True, cache=True)
def find_exact_solution(x: np.ndarray, t: float64, ts: float64):
    u = np.sin(2.0 * np.pi * x) / (2.0 * np.pi * ts)
    diff = u - np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)
    while np.linalg.norm(diff) / diff.size > 1.0e-14:
        u = np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)
        diff = u - np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)

    return u


def run(N, source, bc, ghc, ts, scheme, dt, plot=False):
    phi = create_Var([N + 1], ghc, [(0.0, 1.0)])
    for i in range(phi.shape[0]):
        phi.data[i + phi.ghc, 0, 0] = np.sin(2.0 * np.pi * phi.x_axis[i]) / (2.0 * np.pi * ts)
    periodic_1d(phi)

    t = 0.0
    while t < 0.75 * ts:
        t += dt
        rk3(dt, phi, source, bc, phi.data, scheme)

    phi_exact = create_Var([N + 1], ghc, [(0.0, 1.0)])
    phi_exact.data[ghc:N + ghc + 1, 0, 0] = find_exact_solution(phi_exact.x_axis, t, ts)
    periodic_1d(phi_exact)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(phi.data[:, 0, 0])
        ax.plot(phi_exact.data[:, 0, 0])

    return np.linalg.norm(phi_exact.data[ghc:N + ghc + 1] - phi.data[ghc:N + ghc + 1]) / (N + 1)


if __name__ == "__main__":
    run_scheme = getattr(import_module("scheme.spatial"), input('Choose scheme (CCD, UCCD, WENO_JS, WENO_Z, CRWENO): '))
    data = {}
    for i in range(5, 11):
        data[2 ** i] = run(2 ** i, BurgerEq_source_1d, periodic_1d, 3, 2.0, run_scheme, 0.1 * 1.0 / 2 ** 10)
    find_order(data)
