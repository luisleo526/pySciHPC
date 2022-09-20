from typing import Callable

import numpy as np
from numba import njit, float64, prange

from ....utils.matrix_solver import TDMA
from ....utils.utils import pad

epsilon = 1.0e-8


@njit(float64[:](float64, float64, float64))
def WENO_weights_JS(b1: float, b2: float, b3: float):
    weights = np.zeros(3, dtype='float64')

    a1 = 1.0 / (epsilon + b1) ** 2
    a2 = 6.0 / (epsilon + b2) ** 2
    a3 = 3.0 / (epsilon + b3) ** 2

    weights[0] = a1 / (a1 + a2 + a3)
    weights[1] = a2 / (a1 + a2 + a3)
    weights[2] = a3 / (a1 + a2 + a3)

    return weights


@njit(float64[:](float64, float64, float64))
def WENO_weights_Z(b1: float, b2: float, b3: float):
    weights = np.zeros(3, dtype='float64')

    a1 = 1.0 * (1.0 + abs(b1 - b3) / (epsilon + b1))
    a2 = 6.0 * (1.0 + abs(b1 - b3) / (epsilon + b2))
    a3 = 3.0 * (1.0 + abs(b1 - b3) / (epsilon + b3))

    weights[0] = a1 / (a1 + a2 + a3)
    weights[1] = a2 / (a1 + a2 + a3)
    weights[2] = a3 / (a1 + a2 + a3)

    return weights


@njit
def WENO_p(a: float, b: float, c: float, d: float, e: float, weights: Callable):
    '''
    a : i-1 | b: i | c: i+1 | d: i+2 | e: i+3
    '''

    b1, b2, b3 = WENO_indicators_p(a, b, c, d, e)
    w1, w2, w3 = weights(b1, b2, b3)

    f3 = (- a + 5.0 * b + 2.0 * c) / 6.0
    f2 = (2.0 * b + 5.0 * c - d) / 6.0
    f1 = (11.0 * c - 7.0 * d + 2.0 * e) / 6.0

    return w1 * f1 + w2 * f2 + w3 * f3


@njit(float64[:](float64, float64, float64, float64, float64))
def WENO_indicators_p(a: float, b: float, c: float, d: float, e: float):
    '''
    a : i-1 | b: i | c: i+1 | d: i+2 | e: i+3
    '''
    results = np.zeros(3, dtype='float64')
    results[2] = 13.0 * (a - 2.0 * b + c) ** 2 + 3.0 * (a - 4.0 * b + 3.0 * c) ** 2
    results[1] = 13.0 * (b - 2.0 * c + d) ** 2 + 3.0 * (b - d) ** 2
    results[0] = 13.0 * (c - 2.0 * d + e) ** 2 + 3.0 * (3.0 * c - 4.0 * d + e) ** 2

    return results


@njit
def WENO_m(a: float, b: float, c: float, d: float, e: float, weights: Callable):
    '''
    a : i-2 | b: i-1 | c: i | d: i+1 | e: i+2
    '''
    b1, b2, b3 = WENO_indicators_m(a, b, c, d, e)
    w1, w2, w3 = weights(b1, b2, b3)

    f3 = (- e + 5.0 * d + 2.0 * c) / 6.0
    f2 = (2.0 * d + 5.0 * c - b) / 6.0
    f1 = (11.0 * c - 7.0 * b + 2.0 * a) / 6.0
    return w1 * f1 + w2 * f2 + w3 * f3


@njit(float64[:](float64, float64, float64, float64, float64))
def WENO_indicators_m(a: float, b: float, c: float, d: float, e: float):
    '''
    a : i-2 | b: i-1 | c: i | d: i+1 | e: i+2
    '''
    results = np.zeros(3, dtype='float64')
    results[0] = 13.0 * (a - 2.0 * b + c) ** 2 + 3.0 * (a - 4.0 * b + 3.0 * c) ** 2
    results[1] = 13.0 * (b - 2.0 * c + d) ** 2 + 3.0 * (b - d) ** 2
    results[2] = 13.0 * (c - 2.0 * d + e) ** 2 + 3.0 * (3.0 * c - 4.0 * d + e) ** 2

    return results


@njit(float64[:](float64[:], float64[:], float64), parallel=True, fastmath=True, nogil=True)
def WENO_JS(ff: np.ndarray, c: np.ndarray, dx: float):
    fh = np.zeros_like(ff)
    fx = np.zeros_like(ff)
    f = pad(ff, 3)

    for i in prange(ff.size - 1):
        I = i + 3
        if c[i] + c[i + 1] < 0.0:
            fh[i] = WENO_p(f[I - 1], f[I], f[I + 1], f[I + 2], f[I + 3], WENO_weights_JS)
        else:
            fh[i] = WENO_m(f[I - 2], f[I - 1], f[I], f[I + 1], f[I + 2], WENO_weights_JS)

    for i in prange(1, ff.size):
        fx[i] = (fh[i] - fh[i - 1]) / dx

    return fx


@njit(float64[:](float64[:], float64[:], float64), parallel=True, fastmath=True, nogil=True)
def WENO_Z(ff: np.ndarray, c: np.ndarray, dx: float):
    fh = np.zeros_like(ff)
    fx = np.zeros_like(ff)
    f = pad(ff, 3)

    for i in prange(ff.size - 1):
        I = i + 3
        if c[i] + c[i + 1] < 0.0:
            fh[i] = WENO_p(f[I - 1], f[I], f[I + 1], f[I + 2], f[I + 3], WENO_weights_Z)
        else:
            fh[i] = WENO_m(f[I - 2], f[I - 1], f[I], f[I + 1], f[I + 2], WENO_weights_Z)

    for i in prange(1, ff.size):
        fx[i] = (fh[i] - fh[i - 1]) / dx

    return fx


@njit(float64[:](float64[:], float64[:], float64), parallel=True, fastmath=True, nogil=True)
def CRWENO(f: np.ndarray, c: np.ndarray, dx: float):
    size = f.size - 6
    ghc = 3

    fp = np.zeros_like(f)
    fm = np.zeros_like(f)
    fh = np.zeros_like(f)
    fx = np.zeros_like(f)

    for i in [-1, size - 1]:
        I = i + ghc
        fp[I] = WENO_p(f[I - 1], f[I], f[I + 1], f[I + 2], f[I + 3], WENO_weights_Z)
        fm[I] = WENO_m(f[I - 2], f[I - 1], f[I], f[I + 1], f[I + 2], WENO_weights_Z)

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

        b1, b2, b3 = WENO_indicators_p(f[I - 1], f[I], f[I + 1], f[I + 2], f[I + 3])

        a1 = 2.0 * (1.0 + abs(b1 - b3) / (epsilon + b1))
        a2 = 5.0 * (1.0 + abs(b1 - b3) / (epsilon + b2))
        a3 = 3.0 * (1.0 + abs(b1 - b3) / (epsilon + b3))

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        ap[i] = w3 / 3.0
        bp[i] = (w1 + 2.0 * (w2 + w3)) / 3.0
        cp[i] = (w2 + 2.0 * w1) / 3.0
        sp[i] = (5.0 * w3 + w2) / 6.0 * f[I] + (w3 + 5.0 * (w2 + w1)) / 6.0 * f[I + 1] + w1 / 6.0 * f[I + 2]

        b1, b2, b3 = WENO_indicators_m(f[I - 2], f[I - 1], f[I], f[I + 1], f[I + 2])

        a1 = 2.0 * (1.0 + abs(b1 - b3) / (epsilon + b1))
        a2 = 5.0 * (1.0 + abs(b1 - b3) / (epsilon + b2))
        a3 = 3.0 * (1.0 + abs(b1 - b3) / (epsilon + b3))

        w1 = a1 / (a1 + a2 + a3)
        w2 = a2 / (a1 + a2 + a3)
        w3 = a3 / (a1 + a2 + a3)

        am[i] = (2.0 * w1 + w2) / 3.0
        bm[i] = (w1 + 2.0 * (w2 + w3)) / 3.0
        cm[i] = w3 / 3.0
        sm[i] = w1 / 6.0 * f[I - 1] + (5.0 * (w1 + w2) + w3) / 6.0 * f[I] + (w2 + 5.0 * w3) / 6.0 * f[I + 1]

    sp[0] = sp[0] - ap[0] * fp[ghc - 1]
    sp[-1] = sp[-1] - cp[-1] * fp[size + ghc - 1]
    fp[ghc:size + ghc - 1] = TDMA(ap, bp, cp, sp)

    sm[0] = sm[0] - am[0] * fm[ghc - 1]
    sm[-1] = sm[-1] - cm[-1] * fm[size + ghc - 1]
    fm[ghc:size + ghc - 1] = TDMA(am, bm, cm, sm)

    for i in prange(f.size - 1):
        if c[i] + c[i + 1] > 0.0:
            fh[i] = fm[i]
        else:
            fh[i] = fp[i]

    for i in prange(1, f.size):
        fx[i] = (fh[i] - fh[i - 1]) / dx

    return fx


@njit(float64[:](float64[:], float64[:], float64), parallel=True, fastmath=True, nogil=True)
def CRWENO_LD(f: np.ndarray, c: np.ndarray, dx: float):
    size = f.size - 6
    ghc = 3

    fp = np.zeros_like(f)
    fm = np.zeros_like(f)
    fh = np.zeros_like(f)
    fx = np.zeros_like(f)

    for i in [-1, size - 1]:
        I = i + ghc
        fp[I] = WENO_p(f[I - 1], f[I], f[I + 1], f[I + 2], f[I + 3], WENO_weights_Z)
        fm[I] = WENO_m(f[I - 2], f[I - 1], f[I], f[I + 1], f[I + 2], WENO_weights_Z)

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

        b4 = 13.0 * (f[I - 2] - 2.0 * f[I - 1] + f[I]) ** 2 + 3.0 * (-3.0 * f[I - 2] + 8.0 * f[I - 1] - 5.0 * f[I]) ** 2
        b1, b2, b3 = WENO_indicators_p(f[I - 1], f[I], f[I + 1], f[I + 2], f[I + 3])
        b4 = max(b3, b4)

        a1 = 3.0 / 20.0 * (1.0 + abs(b4 - b1) / (epsilon + b1))
        a2 = 9.0 / 20.0 * (1.0 + abs(b4 - b1) / (epsilon + b2))
        a3 = 7.0 / 20.0 * (1.0 + abs(b4 - b1) / (epsilon + b3))
        a4 = 1.0 / 20.0 * (1.0 + abs(b4 - b1) / (epsilon + b4))

        w1 = a1 / (a1 + a2 + a3 + a4)
        w2 = a2 / (a1 + a2 + a3 + a4)
        w3 = a3 / (a1 + a2 + a3 + a4)
        w4 = a4 / (a1 + a2 + a3 + a4)

        ap[i] = (w3 + 2.0 * w4) / 3.0
        bp[i] = (w1 + 2.0 * (w2 + w3) + w4) / 3.0
        cp[i] = (w2 + 2.0 * w1) / 3.0
        sp[i] = w1 / 6.0 * f[I + 2] + (5.0 * (w1 + w2) + w3) / 6.0 * f[I + 1] + (w2 + 5.0 * (w3 + w4)) / 6.0 * f[I] + \
                w4 / 6.0 * f[I - 1]

        b1, b2, b3 = WENO_indicators_m(f[I - 2], f[I - 1], f[I], f[I + 1], f[I + 2])
        b4 = 13.0 * (f[I + 1] - 2.0 * f[i + 2] + f[I + 3]) ** 2 + 3.0 * (
                -5.0 * f[I + 1] + 8.0 * f[I + 2] - 3.0 * f[I + 3]) ** 2
        b4 = max(b3, b4)

        a1 = 3.0 / 20.0 * (1.0 + abs(b4 - b1) / (epsilon + b1))
        a2 = 9.0 / 20.0 * (1.0 + abs(b4 - b1) / (epsilon + b2))
        a3 = 7.0 / 20.0 * (1.0 + abs(b4 - b1) / (epsilon + b3))
        a4 = 1.0 / 20.0 * (1.0 + abs(b4 - b1) / (epsilon + b4))

        w1 = a1 / (a1 + a2 + a3 + a4)
        w2 = a2 / (a1 + a2 + a3 + a4)
        w3 = a3 / (a1 + a2 + a3 + a4)
        w4 = a4 / (a1 + a2 + a3 + a4)

        am[i] = (2.0 * w1 + w2) / 3.0
        bm[i] = (w1 + 2.0 * (w2 + w3) + w4) / 3.0
        cm[i] = (w3 + 2.0 * w4) / 3.0
        sm[i] = w1 / 6.0 * f[I - 1] + (5.0 * (w1 + w2) + w3) / 6.0 * f[I] + (w2 + 5.0 * (w3 + w4)) / 6.0 * f[I + 1] + \
                w4 / 6.0 * f[I + 2]

    sp[0] = sp[0] - ap[0] * fp[ghc - 1]
    sp[-1] = sp[-1] - cp[-1] * fp[size + ghc - 1]
    fp[ghc:size + ghc - 1] = TDMA(ap, bp, cp, sp)

    sm[0] = sm[0] - am[0] * fm[ghc - 1]
    sm[-1] = sm[-1] - cm[-1] * fm[size + ghc - 1]
    fm[ghc:size + ghc - 1] = TDMA(am, bm, cm, sm)

    for i in prange(f.size - 1):
        if c[i] + c[i + 1] > 0.0:
            fh[i] = fm[i]
        else:
            fh[i] = fp[i]

    for i in prange(1, f.size):
        fx[i] = (fh[i] - fh[i - 1]) / dx

    return fx
