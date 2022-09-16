from typing import Callable

from .solvers import CudaDerivativesSolver
from ..core.data import Scalar, Vector


def solve_hyperbolic(f: Scalar, c: Vector, geo: Scalar, temporal: Callable, bc: Callable, source: Callable,
                     solver: CudaDerivativesSolver, *args):
    temporal(f, geo, c, source, bc, solver, *args)
