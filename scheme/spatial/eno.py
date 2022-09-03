import numpy as np
from numba import njit, float64, int32, prange
from utils.matrix_solver import TDMA


@njit(float64[:, :](float64[:], int32, int32), parallel=True)
def weno_js(f: np.ndarray, size: int, ghc: int):
    assert f.size == size + 2 * ghc

    fp = np.zeros_like(f)
    fm = np.zeros_like(f)
    epsilon = 1.0e-10

    for i in prange(-1, size):
        I = i + ghc

        b3 = 13.0 * (f[I - 1] - 2.0 * f[I] + f[I + 1]) ** 2 + 3.0 * (f[I - 1] - 4.0 * f[I] + 3.0 * f[I + 1]) ** 2
        b2 = 13.0 * (f[I] - 2.0 * f[I + 1] + f[I + 2]) ** 2 + 3.0 * (f[I] - f[I + 2]) ** 2
        b1 = 13.0 * (f[I + 1] - 2.0 * f[I + 2] + f[I + 3]) ** 2 + 3.0 * (
                3.0 * f[I + 1] - 4.0 * f[I + 2] + f[I + 3]) ** 2

        a1 = 1.0 / (epsilon + b1) ** 2
        a2 = 6.0 / (epsilon + b2) ** 2
        a3 = 3.0 / (epsilon + b3) ** 2

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        fp[I] = w3 * (-f[I - 1] + 5.0 * f[I] + 2.0 * f[I + 1]) / 6.0 + \
                w2 * (2.0 * f[I] + 5.0 * f[I + 1] - f[I + 2]) / 6.0 + \
                w1 * (11.0 * f[I + 1] - 7.0 * f[I + 2] + 2.0 * f[I + 3]) / 6.0

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

    return np.stack((fp, fm))


@njit(float64[:, :](float64[:], int32, int32), parallel=True)
def weno_z(f: np.ndarray, size: int, ghc: int):
    assert f.size == size + 2 * ghc

    fp = np.zeros_like(f)
    fm = np.zeros_like(f)
    epsilon = 1.0e-10

    for i in prange(-1, size):
        I = i + ghc

        b3 = 13.0 * (f[I - 1] - 2.0 * f[I] + f[I + 1]) ** 2 + 3.0 * (f[I - 1] - 4.0 * f[I] + 3.0 * f[I + 1]) ** 2
        b2 = 13.0 * (f[I] - 2.0 * f[I + 1] + f[I + 2]) ** 2 + 3.0 * (f[I] - f[I + 2]) ** 2
        b1 = 13.0 * (f[I + 1] - 2.0 * f[I + 2] + f[I + 3]) ** 2 + 3.0 * (
                3.0 * f[I + 1] - 4.0 * f[I + 2] + f[I + 3]) ** 2

        a1 = 1.0 * (1.0 + abs(b1 - b3) / (epsilon + b1))
        a2 = 6.0 * (1.0 + abs(b1 - b3) / (epsilon + b2))
        a3 = 3.0 * (1.0 + abs(b1 - b3) / (epsilon + b3))

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        fp[I] = w3 * (-f[I - 1] + 5.0 * f[I] + 2.0 * f[I + 1]) / 6.0 + \
                w2 * (2.0 * f[I] + 5.0 * f[I + 1] - f[I + 2]) / 6.0 + \
                w1 * (11.0 * f[I + 1] - 7.0 * f[I + 2] + 2.0 * f[I + 3]) / 6.0

        b1 = 13.0 * (f[I - 2] - 2.0 * f[I - 1] + f[I]) ** 2 + 3.0 * (f[I - 2] - 4.0 * f[I - 1] + 3.0 * f[I]) ** 2
        b2 = 13.0 * (f[I - 1] - 2.0 * f[I] + f[I + 1]) ** 2 + 3.0 * (f[I - 1] - f[I + 1]) ** 2
        b3 = 13.0 * (f[I] - 2.0 * f[I + 1] + f[I + 2]) ** 2 + 3.0 * (3.0 * f[I] - 4.0 * f[I + 1] + f[I + 2]) ** 2

        a1 = 1.0 * (1.0 + abs(b1 - b3) / (epsilon + b1))
        a2 = 6.0 * (1.0 + abs(b1 - b3) / (epsilon + b2))
        a3 = 3.0 * (1.0 + abs(b1 - b3) / (epsilon + b3))

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        fm[I] = w1 / 3.0 * f[I - 2] - (7.0 * w1 + w2) / 6.0 * f[I - 1] + \
                (11.0 * w1 + 5.0 * w2 + 2.0_8 * w3) / 6.0 * f[I] + \
                (2.0 * w2 + 5.0 * w3) / 6.0 * f[I + 1] - w3 / 6.0 * f[I + 2]

    return np.stack((fp, fm))


@njit(float64[:, :](float64[:], int32, int32, float64[:]), parallel=True)
def crweno(f: np.ndarray, size: int, ghc: int, coeffs: np.ndarray):
    assert f.size == size + 2 * ghc

    fp = np.zeros_like(f)
    fm = np.zeros_like(f)
    epsilon = 1.0e-10

    for i in [-1, size - 1]:
        I = i + ghc

        b3 = 13.0 * (f[I - 1] - 2.0 * f[I] + f[I + 1]) ** 2 + 3.0 * (f[I - 1] - 4.0 * f[I] + 3.0 * f[I + 1]) ** 2
        b2 = 13.0 * (f[I] - 2.0 * f[I + 1] + f[I + 2]) ** 2 + 3.0 * (f[I] - f[I + 2]) ** 2
        b1 = 13.0 * (f[I + 1] - 2.0 * f[I + 2] + f[I + 3]) ** 2 + 3.0 * (
                3.0 * f[I + 1] - 4.0 * f[I + 2] + f[I + 3]) ** 2

        a1 = 1.0 * (1.0 + abs(b1 - b3) / (epsilon + b1))
        a2 = 6.0 * (1.0 + abs(b1 - b3) / (epsilon + b2))
        a3 = 3.0 * (1.0 + abs(b1 - b3) / (epsilon + b3))

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        fp[I] = w3 * (-f[I - 1] + 5.0 * f[I] + 2.0 * f[I + 1]) / 6.0 + \
                w2 * (2.0 * f[I] + 5.0 * f[I + 1] - f[I + 2]) / 6.0 + \
                w1 * (11.0 * f[I + 1] - 7.0 * f[I + 2] + 2.0 * f[I + 3]) / 6.0

        b1 = 13.0 * (f[I - 2] - 2.0 * f[I - 1] + f[I]) ** 2 + 3.0 * (f[I - 2] - 4.0 * f[I - 1] + 3.0 * f[I]) ** 2
        b2 = 13.0 * (f[I - 1] - 2.0 * f[I] + f[I + 1]) ** 2 + 3.0 * (f[I - 1] - f[I + 1]) ** 2
        b3 = 13.0 * (f[I] - 2.0 * f[I + 1] + f[I + 2]) ** 2 + 3.0 * (3.0 * f[I] - 4.0 * f[I + 1] + f[I + 2]) ** 2

        a1 = 1.0 * (1.0 + abs(b1 - b3) / (epsilon + b1))
        a2 = 6.0 * (1.0 + abs(b1 - b3) / (epsilon + b2))
        a3 = 3.0 * (1.0 + abs(b1 - b3) / (epsilon + b3))

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        fm[I] = w1 / 3.0 * f[I - 2] - (7.0 * w1 + w2) / 6.0 * f[I - 1] + \
                (11.0 * w1 + 5.0 * w2 + 2.0_8 * w3) / 6.0 * f[I] + \
                (2.0 * w2 + 5.0 * w3) / 6.0 * f[I + 1] - w3 / 6.0 * f[I + 2]

    ap = np.zeros((size - 1))
    bp = np.zeros((size - 1))
    cp = np.zeros((size - 1))
    sp = np.zeros((size - 1))

    am = np.zeros((size - 1))
    bm = np.zeros((size - 1))
    cm = np.zeros((size - 1))
    sm = np.zeros((size - 1))

    for i in prange(size - 1):
        I = i + ghc

        b3 = 13.0 * (f[I - 1] - 2.0 * f[I] + f[I + 1]) ** 2 + 3.0 * (f[I - 1] - 4.0 * f[I] + 3.0 * f[I + 1]) ** 2
        b2 = 13.0 * (f[I] - 2.0 * f[I + 1] + f[I + 2]) ** 2 + 3.0 * (f[I] - f[I + 2]) ** 2
        b1 = 13.0 * (f[I + 1] - 2.0 * f[I + 2] + f[I + 3]) ** 2 + 3.0 * (
                3.0 * f[I + 1] - 4.0 * f[I + 2] + f[I + 3]) ** 2

        a1 = coeffs[0] * (1.0 + abs(b1 - b3) / (epsilon + b1))
        a2 = coeffs[1] * (1.0 + abs(b1 - b3) / (epsilon + b2))
        a3 = coeffs[2] * (1.0 + abs(b1 - b3) / (epsilon + b3))

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        ap[i] = w3 / 3.0
        bp[i] = (w1 + 2.0 * (w2 + w3)) / 3.0
        cp[i] = (w2 + 2.0 * w1) / 3.0
        sp[i] = (5.0 * w3 + w2) / 6.0 * f[I] + (w3 + 5.0 * (w2 + w1)) / 6.0 * f[I + 1] + w1 / 6.0 * f[I + 2]

        b1 = 13.0 * (f[I - 2] - 2.0 * f[I - 1] + f[I]) ** 2 + 3.0 * (f[I - 2] - 4.0 * f[I - 1] + 3.0 * f[I]) ** 2
        b2 = 13.0 * (f[I - 1] - 2.0 * f[I] + f[I + 1]) ** 2 + 3.0 * (f[I - 1] - f[I + 1]) ** 2
        b3 = 13.0 * (f[I] - 2.0 * f[I + 1] + f[I + 2]) ** 2 + 3.0 * (3.0 * f[I] - 4.0 * f[I + 1] + f[I + 2]) ** 2

        a1 = coeffs[0] * (1.0 + abs(b1 - b3) / (epsilon + b1))
        a2 = coeffs[1] * (1.0 + abs(b1 - b3) / (epsilon + b2))
        a3 = coeffs[2] * (1.0 + abs(b1 - b3) / (epsilon + b3))

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        am[i] = (2.0 * w1 + w2) / 3.0
        bm[i] = (w1 + 2.0 * (w2 + w3)) / 3.0
        cm[i] = w3 / 3.0
        sm[i] = w1 / 6.0 * f[I - 1] + (5.0 * (w1 + w2) + w3) / 6.0 * f[I] + (w2 + 5.0 * w3) / 6.0 * f[I + 1]

    sp[0] = sp[0] - ap[0] * fp[ghc - 1]
    sp[-1] = sp[-1] - cp[-1] * fp[size + ghc - 1]
    fp[ghc:size+ghc-1] = TDMA(ap, bp, cp, sp)

    sm[0] = sm[0] - am[0] * fm[ghc - 1]
    sm[-1] = sm[-1] - cm[-1] * fm[size + ghc - 1]
    fm[ghc:size + ghc - 1] = TDMA(am, bm, cm, sm)

    return np.stack((fp, fm))
