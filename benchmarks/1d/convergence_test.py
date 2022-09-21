import sys

sys.path.insert(0, '../../')
import time

import numpy as np

from pySciHPC.core import solve_hyperbolic
from pySciHPC.core.boundary_conditions.cell import periodic
from pySciHPC.core.data import Scalar, Vector
from pySciHPC.core.pde_source.convection_equation import pure_convection_source
from pySciHPC.core.scheme.temporal import rk3
from pySciHPC.core.scheme.spatial import UCCD, QUICK, WENO_JS, WENO_Z, CRWENO, UpwindSecondOrder
from pySciHPC.utils.utils import find_order, l2_norm
from numba import config


def run(N, source, bc, ghc, c, scheme, dt):
    geo_dict = dict(_size=[N], ghc=ghc, _axis_data=[(-1.0, 1.0)])

    geo = Scalar(**geo_dict, no_data=True)
    phi = Scalar(**geo_dict, no_axis=True)

    phi.core = np.sin(np.pi * geo.mesh.x - np.sin(np.pi * geo.mesh.x) / np.pi)
    bc(phi.data.cpu[0], phi.ghc, phi.ndim)

    vel = Vector(**geo_dict)
    vel.x.cell.cpu[0] = np.ones_like(phi.data.cpu[0]) * c

    t = 0.0
    cpu_time = -time.time()
    while t < 2.0:
        t += dt
        solve_hyperbolic(phi, vel, rk3, bc, source, dt, scheme)

    error = l2_norm(np.sin(np.pi * (geo.mesh.x - c * t) - np.sin(np.pi * (geo.mesh.x - c * t)) / np.pi), phi.core)
    return error, cpu_time + time.time()


if __name__ == "__main__":

    config.THREADING_LAYER = 'threadsafe'

    data = {}
    for i in range(5, 10):
        data[2 ** i] = run(2 ** i, pure_convection_source, periodic, 3, -1.0, UpwindSecondOrder, 0.1 / 2 ** 9)
        print(2 ** i, data[2 ** i])
    print("---Negative speed---")
    find_order(data)

    data = {}
    for i in range(5, 10):
        data[2 ** i] = run(2 ** i, pure_convection_source, periodic, 3, 1.0, UpwindSecondOrder, 0.1 / 2 ** 9)
        print(2 ** i, data[2 ** i])
    print("---Positive speed---")
    find_order(data)
