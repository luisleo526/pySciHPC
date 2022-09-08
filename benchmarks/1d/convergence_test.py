from importlib import import_module
from typing import Callable
import time
import numpy as np
from numba import njit, int32

from pycfd.boundary_conditions import periodic
from pycfd.functions.derivatives import find_fx
from pycfd.objects import Scalar
from pycfd.scheme.temporal import rk3
from pycfd.utils import find_order, l2_norm


@njit(parallel=True, fastmath=True, nogil=True)
def convection_source_1d(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32, c: np.ndarray, scheme: Callable):
    return - find_fx(f, grids[0], c, scheme) * c


def run(N, source, bc, ghc, c, scheme, dt, plot=False):
    geo_dict = dict(_size=[N], ghc=ghc, _axis_data=[(-1.0, 1.0)])

    geo = Scalar(**geo_dict, no_data=True)
    phi = Scalar(**geo_dict, no_axis=True)

    # phi.core = np.sin(np.pi * geo.mesh.x - np.sin(np.pi * geo.mesh.x) / np.pi)
    phi.core = np.sin(np.pi * geo.mesh.x)
    bc(phi.data.cpu[0], phi.ghc, phi.ndim)

    c_array = np.ones_like(phi.data.cpu[0]) * c

    t = 0.0
    cpu_time = -time.time()
    while t < 2.0:
        t += dt
        phi.data.cpu[0] = rk3(dt, phi.data.cpu[0], geo.grids, phi.ghc, phi.ndim, source, bc, c_array, scheme)

    phi_exact = Scalar(**geo_dict)
    # phi_exact.core = np.sin(
    #     np.pi * (geo.mesh.x - c * (t - 2.0)) - np.sin(np.pi * (geo.mesh.x - c * (t - 2.0))) / np.pi)
    phi_exact.core = np.sin(np.pi * (geo.mesh.x - c * (t - 2.0)))
    bc(phi_exact.data.cpu[0], phi_exact.ghc, phi_exact.ndim)

    error = l2_norm(phi_exact.core, phi.core)
    return error, cpu_time + time.time()


if __name__ == "__main__":
    run_scheme = getattr(import_module("pycfd.scheme.spatial"),
                         input('Choose scheme (CCD, UCCD, WENO_JS, WENO_Z, CRWENO, CRWENO_LD): '))
    data = {}
    for i in range(5, 10):
        data[2 ** i] = run(2 ** i, convection_source_1d, periodic, 3, 1.0, run_scheme, 0.01 * 2.0 / 2 ** 9)
    print("---Positive speed---")
    find_order(data)
    # data = {}
    # for i in range(5, 10):
    #     data[2 ** i] = run(2 ** i, convection_source_1d, periodic, 3, -1.0, run_scheme, 0.01 * 2.0 / 2 ** 9)
    # print("---Negative speed---")
    # find_order(data)
