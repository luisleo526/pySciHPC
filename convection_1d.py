import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from boundary_conditions.periodic import periodic_1d
from methods.find_fx import find_fx_eno
from objects.variable import create_Var, Var
from scheme.spatial.eno import weno_js
from scheme.temporal.runge_kutta import rk3


@njit
def convection_source_1d(f: Var, c: np.ndarray):
    return - find_fx_eno(f, c, weno_js) * c


def run(N, source, bc, ghc):
    phi = create_Var([N + 1], ghc, [(-1.0, 1.0)])
    for i in range(phi.shape[0]):
        phi.data[i + phi.ghc, 0, 0] = np.sin(np.pi * phi.x_axis[i])
    periodic_1d(phi)

    c = -1.0
    c_array = np.ones_like(phi.data) * c
    dt = phi.dx * 0.01

    t = 0.0
    while t < 2.0:
        t += dt
        rk3(dt, phi, source, bc, c_array)

    phi_exact = create_Var([N + 1], ghc, [(-1.0, 1.0)])
    for i in range(phi_exact.shape[0]):
        phi_exact.data[i + phi_exact.ghc, 0, 0] = np.sin(np.pi * (phi_exact.x_axis[i] + c * (t - 2.0)))
    periodic_1d(phi_exact)

    fig, ax = plt.subplots()

    ax.plot(phi.data[:, 0, 0])
    ax.plot(phi_exact.data[:, 0, 0])

    return np.linalg.norm(phi_exact.data[ghc:N + ghc + 1] - phi.data[ghc:N + ghc + 1]) / (N + 1)
