from typing import Callable

import numpy as np
from numba import njit

from objects.variable import Var


@njit
def find_fx_eno(f: Var, c: np.ndarray, eno_scheme: Callable, *args) -> np.ndarray:
    fx = np.zeros_like(f.data)

    for j in range(f.shape[1]):
        for k in range(f.shape[2]):
            fpm = eno_scheme(f.data[:, j, k], f.shape[0], f.ghc, *args)
            fh = np.zeros_like(fpm[0])
            for i in range(f.data.shape[0]):
                if c[i] > 0:
                    fh[i] = fpm[1, i]
                else:
                    fh[i] = fpm[0, i]
            for i in range(f.shape[0]):
                I = i + f.ghc
                fx[I, j, k] = (fh[I] - fh[I - 1]) / f.dx

    return fx
