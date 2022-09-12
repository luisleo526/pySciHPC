import time
from importlib import import_module

import numpy as np

from pySciHPC import solve_hyperbolic
from pySciHPC.boundary_conditions import cuda_periodic
from pySciHPC.objects import Scalar, Vector
from pySciHPC.pde_source.convection_equation import cuda_pure_convection_source
from pySciHPC.scheme.temporal import cuda_rk3
from pySciHPC.utils import find_order, l2_norm
from numba import config


def run(N, source, bc, ghc, c, scheme, dt):
    geo_dict = dict(_size=[N], ghc=ghc, _axis_data=[(-1.0, 1.0)], use_cuda=True, threadsperblock=8)

    geo = Scalar(**geo_dict, no_data=True)
    phi = Scalar(**geo_dict, no_axis=True)

    # phi.core = np.sin(np.pi * geo.mesh.x - np.sin(np.pi * geo.mesh.x) / np.pi)
    phi.core = np.sin(np.pi * geo.mesh.x)
    phi.to_device()
    bc(phi, geo)

    vel = Vector(**geo_dict)
    vel.x.data.cpu[0] = np.ones_like(phi.data.cpu[0]) * c

    phi.to_device()
    vel.to_device()

    t = 0.0
    cpu_time = -time.time()
    while t < 2.0:
        t += dt
        solve_hyperbolic(phi, vel, geo, cuda_rk3, bc, source, dt, scheme)

    phi_exact = Scalar(**geo_dict)
    # phi_exact.core = np.sin(np.pi * (geo.mesh.x - c * t) - np.sin(np.pi * (geo.mesh.x - c * t)) / np.pi)
    phi_exact.core = np.sin(np.pi * (geo.mesh.x - c * t))

    phi.to_host()
    error = l2_norm(phi_exact.core, phi.core)
    with open(f"{N}.data", "w") as f:
        for i in range(phi.core.shape[0]):
            f.write(f"{phi.data.gpu[0, i + ghc, 0, 0]}, {phi.core[i]}, {phi_exact.core[i]}\n")

    return error, cpu_time + time.time()


if __name__ == "__main__":

    config.THREADING_LAYER = 'threadsafe'

    run_scheme = getattr(import_module("pySciHPC.scheme.spatial"),
                         input('Choose scheme (CCD, UCCD, WENO_JS, WENO_Z, CRWENO, CRWENO_LD): '))
    data = {}
    for i in range(5, 10):
        data[2 ** i] = run(2 ** i, cuda_pure_convection_source, cuda_periodic, 3, 1.0, run_scheme, 0.1 / 2 ** 9)
        print(2 ** i, data[2 ** i])
    print("---Positive speed---")
    find_order(data)
