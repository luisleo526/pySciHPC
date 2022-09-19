from typing import Callable

from ...kernels import rk3_1, rk3_2, rk3_3
from ...solvers import CudaDerivativesSolver
from ....core.data import Scalar, Vector


def cuda_rk3(f: Scalar, vel: Vector, source: Callable, boundary_condition: Callable,
             solver: CudaDerivativesSolver, *args):
    source(f, vel, solver, solver.src_buffer0, *args)
    rk3_1(solver.src_buffer0, solver.dt, f.data.gpu[0])
    boundary_condition(f)

    source(f, vel, solver, solver.src_buffer1, *args)
    rk3_2(solver.src_buffer0, solver.src_buffer1, solver.dt, f.data.gpu[0])
    boundary_condition(f)

    source(f, vel, solver, solver.src_buffer2, *args)
    rk3_3(solver.src_buffer0, solver.src_buffer1, solver.src_buffer2, solver.dt, f.data.gpu[0])
    boundary_condition(f)
