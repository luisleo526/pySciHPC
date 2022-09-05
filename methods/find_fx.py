import numpy as np
from numba import njit, float64

from objects.variable import Var
from typing import Callable


@njit(parallel=True, fastmath=True)
def find_fx(f: np.ndarray, dx: float64, c: np.ndarray, scheme: Callable, *args):
    fx = np.zeros_like(f)
    for j in range(f.shape[1]):
        for k in range(f.shape[2]):
            fx[:, j, k] = scheme(f[:, j, k], c[:, j, k], dx, *args)
    return fx
