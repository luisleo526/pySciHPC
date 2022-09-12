import time

import numpy as np

from pySciHPC import solve_hyperbolic
from pySciHPC.boundary_conditions import cuda_periodic
from pySciHPC.objects import Scalar, Vector
from pySciHPC.pde_source.convection_equation import cuda_pure_convection_source
from pySciHPC.objects.coeffs import CCDMatrix
from pySciHPC.scheme.temporal import cuda_rk3
from pySciHPC.utils import find_order, l2_norm
from pySciHPC.scheme.spatial import cuda_WENO_JS, cuda_WENO_Z, cuda_UCCD

from numba import config


def run(N, source, bc, ghc, c, scheme, dt):
    geo_dict = dict(_size=[N], ghc=ghc, _axis_data=[(-1.0, 1.0)], use_cuda=True, threadsperblock=8)

    geo = Scalar(**geo_dict, no_data=True)
    phi = Scalar(**geo_dict, no_axis=True)
    coeff = CCDMatrix(phi.data.cpu[0].shape, geo.grids, use_cuda=True)

    # phi.core = np.sin(np.pi * geo.mesh.x - np.sin(np.pi * geo.mesh.x) / np.pi)
    phi.core = np.sin(np.pi * geo.mesh.x)
    phi.to_device()
    bc(phi, geo)

    vel = Vector(**geo_dict)
    vel.x.data.cpu[0] = np.ones_like(phi.data.cpu[0]) * c
    vel.to_device()

    t = 0.0
    cpu_time = -time.time()
    while t < 2.0:
        t += dt
        solve_hyperbolic(phi, vel, geo, cuda_rk3, bc, source, dt, scheme, coeff)

    phi.to_host()
    error = l2_norm(np.sin(np.pi * (geo.mesh.x - c * t)), phi.core)

    return error, cpu_time + time.time()


if __name__ == "__main__":

    config.THREADING_LAYER = 'threadsafe'

    data = {}
    for i in range(5, 8):
        N = 2 ** i
        data[N] = run(N, cuda_pure_convection_source, cuda_periodic, 3, 1.0, cuda_UCCD, 0.1 / 2 ** 7)
        print(N, data[N])
    print("---Positive speed---")
    find_order(data)
