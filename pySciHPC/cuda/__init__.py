from typing import Callable

from .solvers import CudaDerivativesSolver
from ..core.data import Scalar, Vector


def solve_hyperbolic(f: Scalar, c: Vector, temporal: Callable, bc: Callable, source: Callable,
                     solver: CudaDerivativesSolver, *args):
    temporal(f, c, source, bc, solver, *args)
