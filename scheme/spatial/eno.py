import numpy as np
from numba import njit, float64, int32, prange


@njit(float64[:](float64[:], int32, int32), parallel=True, nogil=True)
def weno_js_p(f: np.ndarray, size: int, ghc: int):
    assert f.size == size + 2 * ghc

    fp = np.zeros_like(f)
    epsilon = 1.0e-10

    for i in prange(-1, size):
        I = i + ghc

        b3 = 13.0 * (f[I - 1] - 2.0 * f[I] + f[I + 1]) ** 2 + 3.0 * (f[I - 1] - 4.0 * f[I] + 3.0 * f[I + 1]) ** 2
        b2 = 13.0 * (f[I] - 2.0 * f[I + 1] + f[I + 2]) ** 2 + 3.0 * (f[I] - f[I + 2]) ** 2
        b1 = 13.0 * (f[I + 1] - 2.0 * f[I + 2] + f[I + 3]) ** 2 + 3.0 * (3.0 * f[I + 1] - 4.0 * f[I + 2] + f[I + 3]) ** 2

        a1 = 1.0 / (epsilon + b1) ** 2
        a2 = 6.0 / (epsilon + b2) ** 2
        a3 = 3.0 / (epsilon + b3) ** 2

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        fp[I] = w3 * (-f[I - 1] + 5.0 * f[I] + 2.0 * f[I + 1]) / 6.0 + \
                w2 * (2.0 * f[I] + 5.0 * f[I + 1] - f[I + 2]) / 6.0 + \
                w1 * (11.0 * f[I + 1] - 7.0 * f[I + 2] + 2.0 * f[I + 3]) / 6.0

    return fp


@njit(float64[:](float64[:], int32, int32), parallel=True)
def weno_js_m(f: np.ndarray, size: int, ghc: int):
    assert f.size == size + 2 * ghc

    fm = np.zeros_like(f)
    epsilon = 1.0e-10

    for i in prange(-1, size):
        I = i + ghc

        b1 = 13.0 * (f[I - 2] - 2.0 * f[I - 1] + f[I]) ** 2 + 3.0 * (f[I - 2] - 4.0 * f[I - 1] + 3.0 * f[I]) ** 2
        b2 = 13.0 * (f[I - 1] - 2.0 * f[I] + f[I + 1]) ** 2 + 3.0 * (f[I - 1] - f[I + 1]) ** 2
        b3 = 13.0 * (f[I] - 2.0 * f[I + 1] + f[I + 2]) ** 2 + 3.0 * (3.0 * f[I] - 4.0 * f[I + 1] + f[I + 2]) ** 2

        a1 = 1.0 / (epsilon + b1) ** 2
        a2 = 6.0 / (epsilon + b2) ** 2
        a3 = 3.0 / (epsilon + b3) ** 2

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        fm[I] = w1 / 3.0 * f[I - 2] - (7.0 * w1 + w2) / 6.0 * f[I - 1] + \
                (11.0 * w1 + 5.0 * w2 + 2.0_8 * w3) / 6.0 * f[I] + \
                (2.0 * w2 + 5.0 * w3) / 6.0 * f[I + 1] - w3 / 6.0 * f[I + 2]

    return fm
