import sys

sys.path.insert(0, '../../../')
import time

import cupy as cp
import numpy as np
from numba import config
from numba import float64, njit

from pySciHPC.cuda import solve_hyperbolic
from pySciHPC.cuda.boundary_conditions.periodic import cuda_periodic
from pySciHPC.cuda.solvers import CudaUCCDSovler, CudaDerivativesSolver
from pySciHPC.objects import Scalar, Vector
from pySciHPC.cuda.scheme.temporal import cuda_rk3
from pySciHPC.core.utils import find_order, l2_norm

burger_flux = cp.ElementwiseKernel('float64 f', 'float64 ff', 'ff = f * f * 0.5', 'burger_flux')
neg_assign = cp.ElementwiseKernel('float64 f', 'float64 ff', 'ff = -f', 'neg_assign', no_return=True)


def cuda_inviscid_burger(f: Scalar, geo: Scalar, vel: Vector, solver: CudaDerivativesSolver, s: cp.ndarray,
                         *args):
    neg_assign(solver.find_fx(burger_flux(f.data.gpu[0], solver.sol_buffer0), vel.x.data.gpu[0]), s)


@njit(float64[:](float64[:], float64, float64), parallel=True, fastmath=True, nogil=True)
def find_exact_solution(x: np.ndarray, t: float64, ts: float64):
    u = np.sin(2.0 * np.pi * x) / (2.0 * np.pi * ts)
    diff = u - np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)
    while np.linalg.norm(diff) / diff.size > 1.0e-14:
        u = np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)
        diff = u - np.sin(2.0 * np.pi * (x - u * t)) / (2.0 * np.pi * ts)

    return u


def run(N, source, bc, ghc, ts, scheme, dt):
    geo_dict = dict(_size=[N], ghc=ghc, _axis_data=[(0.0, 1.0)], use_cuda=True, threadsperblock=32)

    geo = Scalar(**geo_dict, no_data=True)
    phi = Scalar(**geo_dict, no_axis=True)
    solver = scheme(phi.data.cpu[0].shape, geo.grids, dt, phi.ndim, phi.blockspergrid, phi.threadsperblock)

    phi.core = np.sin(2.0 * np.pi * geo.mesh.x) / (2.0 * np.pi * ts)
    phi.to_device()
    bc(phi, geo)

    vel = Vector(**geo_dict)
    vel.x.data.gpu[0] = phi.data.gpu[0]

    t = 0.0
    cpu_time = -time.time()
    while t < 0.75 * ts:
        t += dt
        solve_hyperbolic(phi, vel, geo, cuda_rk3, bc, source, solver)

    phi.to_host()

    error = l2_norm(find_exact_solution(geo.mesh.x, t, ts), phi.core)

    return error, cpu_time + time.time()


if __name__ == "__main__":
    config.THREADING_LAYER = 'threadsafe'

    data = {}
    for i in range(5, 11):
        N = 2 ** i
        data[N] = run(N, cuda_inviscid_burger, cuda_periodic, 3, 2.0, CudaUCCDSovler, 0.1 / 2 ** 10)
        print(N, data[N])
    print("---Positive speed---")
    find_order(data)
