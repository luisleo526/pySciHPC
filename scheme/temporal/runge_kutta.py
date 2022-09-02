from typing import Callable

from numba import njit, prange

from objects.variable import Var


@njit(parallel=True)
def rk3(dt: float, f: Var, find_source: Callable, boundary_condition: Callable, *args):
    s1 = find_source(f, *args)
    for i in prange(f.data.shape[0]):
        for j in range(f.data.shape[1]):
            for k in range(f.data.shape[2]):
                f.data[i, j, k] += s1[i, j, k] * dt
    boundary_condition(f)

    s2 = find_source(f, *args)
    for i in prange(f.data.shape[0]):
        for j in range(f.data.shape[1]):
            for k in range(f.data.shape[2]):
                f.data[i, j, k] += (-3.0 * s1[i, j, k] + s2[i, j, k]) / 4.0 * dt
    boundary_condition(f)

    s3 = find_source(f, *args)
    for i in prange(f.data.shape[0]):
        for j in range(f.data.shape[1]):
            for k in range(f.data.shape[2]):
                f.data[i, j, k] += (-s1[i, j, k] - s2[i, j, k] + 8.0 * s3[i, j, k]) / 12.0 * dt
    boundary_condition(f)
