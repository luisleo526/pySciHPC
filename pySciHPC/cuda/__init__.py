from typing import Callable

from pySciHPC.core.data import Scalar, Vector
from pySciHPC.cuda.solvers import CudaDerivativesSolver


def solve_hyperbolic(f: Scalar, c: Vector, geo: Scalar, temporal: Callable, bc: Callable, source: Callable,
                     solver: CudaDerivativesSolver, *args):
    temporal(f, geo, c, source, bc, solver, *args)
