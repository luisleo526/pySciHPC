from typing import Callable

from pySciHPC import Scalar, Vector
from pySciHPC.cuda.solvers import CudaDerivativesSolver
from pySciHPC.cuda.kernels import rk3_1, rk3_2, rk3_3


def cuda_rk3(f: Scalar, geo: Scalar, vel: Vector, source: Callable, boundary_condition: Callable,
             solver: CudaDerivativesSolver, *args):
    source(f, geo, vel, solver, solver.src_buffer0, *args)
    rk3_1(solver.src_buffer0, solver.dt, f.data.gpu[0])
    boundary_condition(f, geo)

    source(f, geo, vel, solver, solver.src_buffer1, *args)
    rk3_2(solver.src_buffer0, solver.src_buffer1, solver.dt, f.data.gpu[0])
    boundary_condition(f, geo)

    source(f, geo, vel, solver, solver.src_buffer2, *args)
    rk3_3(solver.src_buffer0, solver.src_buffer1, solver.src_buffer2, solver.dt, f.data.gpu[0])
    boundary_condition(f, geo)
