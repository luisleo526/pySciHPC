from typing import Callable, Union, Optional

import numpy as np

from pySciHPC.boundary_conditions.zero_order import zero_order
from pySciHPC.functions.gradients import Godunov_WENO_grad
from pySciHPC.functions.level_set import Delta, Sign
from pySciHPC.objects.base import Scalar, Vector
from pySciHPC.objects.level_set_function import LevelSetFunction
from pySciHPC.utils import l2_norm


def solve_hyperbolic(f: Scalar, c: Union[Vector, np.ndarray], geo: Scalar, temporal: Callable, bc: Callable,
                     source: Callable, dt: float, *args):
    if not f.use_cuda:
        if type(c) is Vector:
            vel = c.of(0)
        else:
            vel = c
        f.data.cpu[0] = temporal(dt, f.data.cpu[0], geo.grids, geo.ghc, geo.ndim, source, bc, vel, *args)
    else:
        assert type(c) is Vector
        temporal(dt, f, geo, c, source, bc, *args)


def solve_hyperbolic_steady(f: Scalar, c: Union[Vector, np.ndarray], geo: Scalar, temporal: Callable, bc: Callable,
                            source: Callable, dt: float, init_func: Optional[Callable], criterion: Callable,
                            tol: float, period: float, *args):
    if not f.use_cuda:

        if type(c) is Vector:
            vel = c.of(0)
        else:
            vel = c

        if init_func is not None:
            f.data.cpu[0], vel = init_func(f.data.cpu[0], geo.grids, geo.ghc, geo.ndim, vel, *args)

        t = 0
        while True:
            t += dt
            tmp = np.copy(f.data.cpu[0])
            f.data.cpu[0] = temporal(dt, f.data.cpu[0], geo.grids, geo.ghc, geo.ndim, source, bc, vel, *args)
            if criterion(f.data.cpu[0], tmp, t, tol, period, geo.grids, geo.ghc, geo.ndim, *args):
                break
