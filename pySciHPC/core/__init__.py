from typing import Callable


def solve_hyperbolic(scalar, vector, temporal: Callable, bc: Callable, source: Callable, dt: float, *args):
    scalar.data.cpu[0] = temporal(dt, scalar.data.cpu[0], scalar.grids, scalar.ghc, scalar.ndim, source, bc,
                                  vector.cell_of0, *args)
