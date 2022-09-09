import time
from importlib import import_module

import numpy as np

from pySciHPC import solve_hyperbolic
from pySciHPC.boundary_conditions import periodic
from pySciHPC.objects import Scalar, Vector
from pySciHPC.pde_source.convection_equation import pure_convection_source
from pySciHPC.scheme.temporal import rk3
from pySciHPC.utils import find_order, l2_norm


def run(N, source, bc, ghc, c, scheme, dt):
    geo_dict = dict(_size=[N], ghc=ghc, _axis_data=[(-1.0, 1.0)])

    geo = Scalar(**geo_dict, no_data=True)
    phi = Scalar(**geo_dict, no_axis=True)

    # phi.core = np.sin(np.pi * geo.mesh.x - np.sin(np.pi * geo.mesh.x) / np.pi)
    phi.core = np.sin(np.pi * geo.mesh.x)
    bc(phi.data.cpu[0], phi.ghc, phi.ndim)

    vel = Vector(**geo_dict)
    vel.x.data.cpu[0] = np.ones_like(phi.data.cpu[0]) * c

    t = 0.0
    cpu_time = -time.time()
    while t < 2.0:
        t += dt
        solve_hyperbolic(phi, vel, geo, rk3, bc, source, dt, scheme)

    phi_exact = Scalar(**geo_dict)
    # phi_exact.core = np.sin(np.pi * (geo.mesh.x - c * t) - np.sin(np.pi * (geo.mesh.x - c * t)) / np.pi)
    phi_exact.core = np.sin(np.pi * (geo.mesh.x - c * t))
    bc(phi_exact.data.cpu[0], phi_exact.ghc, phi_exact.ndim)

    error = l2_norm(phi_exact.core, phi.core)
    return error, cpu_time + time.time()


if __name__ == "__main__":
    run_scheme = getattr(import_module("pySciHPC.scheme.spatial"),
                         input('Choose scheme (CCD, UCCD, WENO_JS, WENO_Z, CRWENO, CRWENO_LD): '))
    data = {}
    for i in range(5, 10):
        data[2 ** i] = run(2 ** i, pure_convection_source, periodic, 3, 1.0, run_scheme, 0.1 / 2 ** 9)
    print("---Positive speed---")
    find_order(data)
