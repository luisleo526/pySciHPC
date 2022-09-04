from typing import Callable

from numba import njit

from objects.variable import Var


@njit(parallel=True, fastmath=True)
def rk3(dt: float, f: Var, find_source: Callable, boundary_condition: Callable, *args):
    s1 = find_source(f, *args)
    f.data += s1 * dt
    boundary_condition(f)

    s2 = find_source(f, *args)
    f.data += (-3.0 * s1 + s2) / 4.0 * dt
    boundary_condition(f)

    s3 = find_source(f, *args)
    f.data += (-s1 - s2 + 8.0 * s3) / 12.0 * dt
    boundary_condition(f)
