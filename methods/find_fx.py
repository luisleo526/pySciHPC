import numpy as np
from numba import njit

from objects.variable import Var
from typing import Callable


@njit(parallel=True, fastmath=True)
def find_fx(f: Var, c: np.ndarray, scheme: Callable, *args):
    fx = np.zeros_like(f.data)
    for j in range(f.shape[1]):
        for k in range(f.shape[2]):
            fx[:, j, k] = scheme(f.data[:, j, k], c[:, j, k], f.dx, *args)
    return fx
