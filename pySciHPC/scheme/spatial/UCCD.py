import numpy as np
from numba import njit, float64, int32, prange

from pySciHPC.utils.matrix_solver import twin_dec, twin_bks
from .CCD import CCD_coeffs_bc


@njit(float64[:, :, :](int32, float64), parallel=True, fastmath=True, cache=True)
def UCCD_coeffs(N: int32, dx: float64):
    a1: float64 = 0.875
    b1: float64 = 0.1251282341599089
    b2: float64 = -0.2487176584009104
    b3: float64 = 0.0001282341599089

    AU = np.zeros((3, N), dtype='float64')
    BU = np.zeros((3, N), dtype='float64')
    AAU = np.zeros((3, N), dtype='float64')
    BBU = np.zeros((3, N), dtype='float64')

    AD = np.zeros((3, N), dtype='float64')
    BD = np.zeros((3, N), dtype='float64')
    AAD = np.zeros((3, N), dtype='float64')
    BBD = np.zeros((3, N), dtype='float64')

    for i in prange(N):
        AU[0, i] = a1
        AU[1, i] = 1.0
        AU[2, i] = 0.0

        AD[0, i] = 0.0
        AD[1, i] = 1.0
        AD[2, i] = a1

        BU[0, i] = b1 * dx
        BU[1, i] = b2 * dx
        BU[2, i] = b3 * dx

        BD[0, i] = -b3 * dx
        BD[1, i] = -b2 * dx
        BD[2, i] = -b1 * dx

        AAU[0, i] = -9.0 / 8.0 / dx
        AAU[1, i] = 0.0
        AAU[2, i] = 9.0 / 8.0 / dx

        AAD[0, i] = -9.0 / 8.0 / dx
        AAD[1, i] = 0.0
        AAD[2, i] = 9.0 / 8.0 / dx

        BBU[0, i] = -1.0 / 8.0
        BBU[1, i] = 1.0
        BBU[2, i] = -1.0 / 8.0

        BBD[0, i] = -1.0 / 8.0
        BBD[1, i] = 1.0
        BBD[2, i] = -1.0 / 8.0

    CCD_coeffs_bc(AU, BU, AAU, BBU, dx)
    CCD_coeffs_bc(AD, BD, AAD, BBD, dx)

    return np.stack((AU, BU, AAU, BBU, AD, BD, AAD, BBD))


@njit(float64[:, :](float64[:], int32, float64), parallel=True, fastmath=True, cache=True)
def UCCD_src(f: np.ndarray, N: int32, dx: float64):
    SU = np.zeros(N, dtype='float64')
    SSU = np.zeros(N, dtype='float64')

    SD = np.zeros(N, dtype='float64')
    SSD = np.zeros(N, dtype='float64')

    SU[0] = (-3.5 * f[0] + 4.0 * f[1] - 0.5 * f[2]) / dx
    SSU[0] = (34.0 / 3.0 * f[0] - 83.0 / 4.0 * f[1] + 10.0 * f[2] - 7.0 / 12.0 * f[3]) / dx ** 2

    SU[-1] = -(-3.5 * f[-1] + 4.0 * f[-2] - 0.5 * f[-3]) / dx
    SSU[-1] = (34.0 / 3.0 * f[-1] - 83.0 / 4.0 * f[-2] + 10.0 * f[-3] - 7.0 / 12.0 * f[-4]) / dx ** 2

    SD[0] = (-3.5 * f[0] + 4.0 * f[1] - 0.5 * f[2]) / dx
    SSD[0] = (34.0 / 3.0 * f[0] - 83.0 / 4.0 * f[1] + 10.0 * f[2] - 7.0 / 12.0 * f[3]) / dx ** 2

    SD[-1] = -(-3.5 * f[-1] + 4.0 * f[-2] - 0.5 * f[-3]) / dx
    SSD[-1] = (34.0 / 3.0 * f[-1] - 83.0 / 4.0 * f[-2] + 10.0 * f[-3] - 7.0 / 12.0 * f[-4]) / dx ** 2

    c3: float64 = -0.06096119008109
    c2: float64 = 1.99692238016218
    c1: float64 = -1.93596119008109

    for i in prange(1, N - 1):
        SU[i] = (c1 * f[i - 1] + c2 * f[i] + c3 * f[i + 1]) / dx
        SSU[i] = (3.0 * f[i - 1] - 6.0 * f[i] + 3.0 * f[i + 1]) / dx ** 2

        SD[i] = -(c3 * f[i - 1] + c2 * f[i] + c1 * f[i + 1]) / dx
        SSD[i] = (3.0 * f[i - 1] - 6.0 * f[i] + 3.0 * f[i + 1]) / dx ** 2

    return np.stack((SU, SSU, SD, SSD))


@njit(float64[:](float64[:], float64[:], float64), parallel=True, fastmath=True, cache=True)
def UCCD(f: np.ndarray, c: np.ndarray, dx: float64):
    AU, BU, AAU, BBU, AD, BD, AAD, BBD = UCCD_coeffs(f.size, dx)

    twin_dec(AU, BU, AAU, BBU)
    twin_dec(AD, BD, AAD, BBD)

    SU, SSU, SD, SSD = UCCD_src(f, f.size, dx)

    fxu, fxxu = twin_bks(AU, BU, AAU, BBU, SU, SSU)
    fxd, fxxd = twin_bks(AD, BD, AAD, BBD, SD, SSD)

    fx = np.zeros_like(fxu)
    fxx = np.zeros_like(fxxu)
    for i in prange(f.size):
        if c[i] > 0.0:
            fx[i] = fxu[i]
        else:
            fx[i] = fxd[i]
        fxx[i] = 0.5 * (fxxu[i] + fxxd[i])

    return fx


@njit(float64[:, :](float64[:], float64[:], float64), parallel=True, fastmath=True, cache=True)
def UCCD_full(f: np.ndarray, c: np.ndarray, dx: float64):
    AU, BU, AAU, BBU, AD, BD, AAD, BBD = UCCD_coeffs(f.size, dx)
    SU, SSU, SD, SSD = UCCD_src(f, f.size, dx)

    fxu, fxxu = twin_bks(AU, BU, AAU, BBU, SU, SSU)
    fxd, fxxd = twin_bks(AD, BD, AAD, BBD, SD, SSD)

    fx = np.zeros_like(fxu)
    fxx = np.zeros_like(fxxu)
    for i in prange(f.size):
        if c[i] > 0.0:
            fx[i] = fxu[i]
        else:
            fx[i] = fxd[i]
        fxx[i] = 0.5 * (fxxu[i] + fxxd[i])

    return np.stack((fx, fxx))
