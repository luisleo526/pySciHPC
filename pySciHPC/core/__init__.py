from typing import Callable, Optional

import numpy as np

from pySciHPC.core.boundary_conditions.zero_order import zero_order
from pySciHPC.core.utils import l2_norm
from pySciHPC.objects.base import Scalar, Vector


def solve_hyperbolic(f: Scalar, c: Vector, geo: Scalar, temporal: Callable, bc: Callable,
                     source: Callable, dt: float, *args):
    f.data.cpu[0] = temporal(dt, f.data.cpu[0], geo.grids, geo.ghc, geo.ndim, source, bc, c.of(0), *args)


def solve_hyperbolic_steady(f: Scalar, c: Vector, geo: Scalar, temporal: Callable, bc: Callable,
                            source: Callable, dt: float, init_func: Optional[Callable], criterion: Callable,
                            tol: float, period: float, *args):
    if init_func is not None:
        f.data.cpu[0], vel = init_func(f.data.cpu[0], geo.grids, geo.ghc, geo.ndim, c.of(0), *args)

    t = 0
    while True:
        t += dt
        tmp = np.copy(f.data.cpu[0])
        f.data.cpu[0] = temporal(dt, f.data.cpu[0], geo.grids, geo.ghc, geo.ndim, source, bc, c.of(0), *args)
        if criterion(f.data.cpu[0], tmp, t, tol, period, geo.grids, geo.ghc, geo.ndim, *args):
            break
