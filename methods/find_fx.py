import numpy as np
from numba import njit

from objects.variable import Var
from scheme.spatial.eno import weno_js_p, weno_js_m


@njit
def find_fx_wenojs(f: Var, c: np.ndarray) -> np.ndarray:
    fx = np.zeros_like(f.data)

    for j in range(f.shape[1]):
        for k in range(f.shape[2]):
            fp = weno_js_p(f.data[:, j, k], f.shape[0], f.ghc)
            fm = weno_js_m(f.data[:, j, k], f.shape[0], f.ghc)
            fh = np.zeros_like(fp)
            for i in range(-1, f.shape[0]):
                I = i + f.ghc
                if c[I] > 0:
                    fh[I] = fm[I]
                else:
                    fh[I] = fp[I]
            for i in range(f.shape[0]):
                I = i + f.ghc
                fx[I, j, k] = (fh[I] - fh[I - 1]) / f.dx

    return fx
