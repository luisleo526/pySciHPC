import cupy as cp

from pySciHPC import Scalar, Vector
from pySciHPC.cuda.solvers import CudaDerivativesSolver
from pySciHPC.cuda.kernels import neg_multi_sum_init, neg_multi_sum


def cuda_pure_convection_source(f: Scalar, geo: Scalar, vel: Vector, solver: CudaDerivativesSolver, s: cp.ndarray,
                                *args):
    neg_multi_sum_init(solver.find_fx(f.data.gpu[0], vel.x.data.gpu[0]), vel.x.data.gpu[0], s)
    if f.ndim > 1:
        neg_multi_sum(solver.find_fy(f.data.gpu[0], vel.y.data.gpu[0]), *vel.y.data.gpu[0], s)
    if f.ndim > 2:
        neg_multi_sum(solver.find_fz(f.data.gpu[0], vel.z.data.gpu[0]), *vel.z.data.gpu[0], s)
