import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from boundary_conditions.periodic import periodic_1d
from methods.find_fx import find_fx_wenojs
from objects.variable import create_Var, Var
from scheme.temporal.runge_kutta import rk3


@njit
def convection_source_1d(f: Var, c: np.ndarray):
    return - find_fx_wenojs(f, c) * c


def run(N, source, bc):
    phi = create_Var([N + 1], 3, [(-1.0, 1.0)])
    for i in range(phi.size):
        phi.data[i + phi.ghc] = np.sin(np.pi * (phi.x_axis[i]))
    periodic_1d(phi)

    c = 1.0
    c_array = np.zeros_like(phi.data) + c
    dt = phi.dx * 0.01

    t = 0.0
    while t < 2.0:
        t += dt
        rk3(dt, phi, source, bc, c_array)

    phi_exact = create_Var([N + 1], 3, [(-1.0, 1.0)])
    for i in range(phi_exact.size):
        phi_exact.data[i + phi_exact.ghc] = np.sin(np.pi * (phi_exact.x_axis[i] + c * (t - 2.0)))
    periodic_1d(phi_exact)

    fig, ax = plt.subplots()

    ax.plot(phi.data[:, 0, 0])
    ax.plot(phi_exact.data[:,0,0])

    return np.linalg.norm(phi_exact.data - phi.data) / phi.size


