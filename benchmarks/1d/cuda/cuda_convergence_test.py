import sys

sys.path.insert(0, '../../../')
import time

import numpy as np
from numba import config

from pySciHPC.cuda import solve_hyperbolic
from pySciHPC.cuda.boundary_conditions.periodic import cuda_periodic
from pySciHPC.cuda.solvers import CudaUCCDSovler, CudaWENOSolver
from pySciHPC.core.data import Scalar, Vector
from pySciHPC.cuda.pde_source.convection import cuda_pure_convection_source
from pySciHPC.cuda.scheme.temporal.Runge_Kutta import cuda_rk3
from pySciHPC.utils.utils import find_order, l2_norm


def run(N, source, bc, ghc, c, scheme, dt):
    geo_dict = dict(_size=[N], ghc=ghc, _axis_data=[(-1.0, 1.0)], use_cuda=True, threadsperblock=32)

    geo = Scalar(**geo_dict, no_data=True)
    phi = Scalar(**geo_dict, no_axis=True)
    solver = scheme(phi.data.cpu[0].shape, geo.grids, dt, phi.ndim, phi.blockspergrid, phi.threadsperblock)

    # phi.core = np.sin(np.pi * geo.mesh.x - np.sin(np.pi * geo.mesh.x) / np.pi)
    phi.core = np.sin(np.pi * geo.mesh.x)
    phi.to_device()
    bc(phi)

    vel = Vector(**geo_dict)
    vel.x.cell.cpu[0] = np.ones_like(phi.data.cpu[0]) * c
    vel.to_device()

    t = 0.0
    cpu_time = -time.time()
    while t < 2.0:
        t += dt
        solve_hyperbolic(phi, vel, cuda_rk3, bc, source, solver)
        # print(t)

    phi.to_host()
    error = l2_norm(np.sin(np.pi * (geo.mesh.x - c * t)), phi.core)

    return error, cpu_time + time.time()


if __name__ == "__main__":

    config.THREADING_LAYER = 'threadsafe'

    data = {}
    for i in range(5, 10):
        N = 2 ** i
        data[N] = run(N, cuda_pure_convection_source, cuda_periodic, 3, 1.0, CudaWENOSolver, 0.1 / 2 ** 9)
        print(N, data[N])
    print("---Positive speed---")
    find_order(data)

    data = {}
    for i in range(5, 10):
        N = 2 ** i
        data[N] = run(N, cuda_pure_convection_source, cuda_periodic, 3, -1.0, CudaWENOSolver, 0.1 / 2 ** 9)
        print(N, data[N])
    print("---Negative speed---")
    find_order(data)
