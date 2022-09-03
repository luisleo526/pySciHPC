from typing import Callable

from boundary_conditions.periodic import periodic_1d
from methods.find_fx import find_fx_eno
from objects.variable import create_Var, Var
from scheme.spatial.eno import *
from scheme.temporal.runge_kutta import rk3


@njit
def convection_source_1d(f: Var, c: np.ndarray, scheme: Callable, *args):
    return - find_fx_eno(f, c, scheme, *args) * c


def run(N, source, bc, ghc):
    phi = create_Var([N + 1], ghc, [(-1.0, 1.0)])
    for i in range(phi.shape[0]):
        phi.data[i + phi.ghc, 0, 0] = np.sin(np.pi * phi.x_axis[i])
    periodic_1d(phi)

    c = -1.0
    c_array = np.ones_like(phi.data) * c
    dt = phi.dx * 0.1

    ocrweno_coeffs = np.array([0.2089141306, 0.4999999998, 0.2910858692], dtype='float64')

    t = 0.0
    while t < 2.0:
        t += dt
        rk3(dt, phi, source, bc, c_array, crweno, ocrweno_coeffs)

    phi_exact = create_Var([N + 1], ghc, [(-1.0, 1.0)])
    for i in range(phi_exact.shape[0]):
        phi_exact.data[i + phi_exact.ghc, 0, 0] = np.sin(np.pi * (phi_exact.x_axis[i] + c * (t - 2.0)))
    periodic_1d(phi_exact)

    return np.linalg.norm(phi_exact.data[ghc + 1:N + ghc] - phi.data[ghc + 1:N + ghc]) / (N - 1)
