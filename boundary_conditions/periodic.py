from numba import njit

from objects.variable import Var


@njit
def periodic_1d(f: Var):
    for i in range(f.ghc):
        f.data[i] = f.data[f.shape[0] + i - 1]
        f.data[f.shape[0] + f.ghc + i] = f.data[f.ghc + i + 1]

