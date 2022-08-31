from objects.variable import Var
import numpy as np


def periodic_1d(f: Var):
    assert f.data.size == f.data.shape[0]

    for i in range(f.ghc):
        f.data[i] = f.data[f.data.shape[0] + i - 1]
        f.data[f.data.shape[0] + f.ghc + i] = f.data[f.ghc + i + 1]
