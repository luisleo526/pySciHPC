from typing import Callable


def solve_hyperbolic(scalar, vector, geo, temporal: Callable, bc: Callable, source: Callable, dt: float, *args):
    scalar.data.cpu[0] = temporal(dt, scalar.data.cpu[0], geo.grids, geo.ghc, geo.ndim, source, bc, vector.of0, *args)
