from pySciHPC.utils.matrix_solver import twin_dec, twin_bks
from numba import njit, float64, int32, prange
import numpy as np


@njit((float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64), cache=True)
def CCD_coeffs_bc(A: np.ndarray, B: np.ndarray, AA: np.ndarray, BB: np.ndarray, dx: float64):
    # Boundary condition -- left
    A[1, 0] = 1.0
    B[1, 0] = 0.0

    AA[1, 0] = 0.0
    BB[1, 0] = 1.0

    A[2, 0] = 2.0
    B[2, 0] = -dx

    AA[2, 0] = -2.5 / dx
    BB[2, 0] = 8.5

    # Boundary condition -- right
    A[1, -1] = 1.0
    B[1, -1] = 0.0

    AA[1, -1] = 0.0
    BB[1, -1] = 1.0

    A[0, -1] = 2.0
    B[0, -1] = dx

    AA[0, -1] = 2.5 / dx
    BB[0, -1] = 8.5


@njit(float64[:, :, :](int32, float64), parallel=True, fastmath=True, cache=True)
def CCD_coeffs(N: int32, dx: float64):
    A = np.zeros((3, N), dtype='float64')
    B = np.zeros((3, N), dtype='float64')
    AA = np.zeros((3, N), dtype='float64')
    BB = np.zeros((3, N), dtype='float64')

    for i in prange(N):
        A[0, i] = 7.0 / 16.0
        A[1, i] = 1.0
        A[2, i] = 7.0 / 16.0

        B[0, i] = dx / 16.0
        B[1, i] = 0.0
        B[2, i] = -dx / 16.0

        AA[0, i] = -9.0 / 8.0 / dx
        AA[1, i] = 0.0
        AA[2, i] = 9.0 / 8.0 / dx

        BB[0, i] = -1.0 / 8.0
        BB[1, i] = 1.0
        BB[2, i] = -1.0 / 8.0

    CCD_coeffs_bc(A, B, AA, BB, dx)
    twin_dec(A, B, AA, BB)

    return np.stack((A, B, AA, BB))


@njit(float64[:, :](float64[:], int32, float64), parallel=True, fastmath=True, cache=True)
def CCD_src(f: np.ndarray, N: int32, dx: float64):
    S = np.zeros(N, dtype='float64')
    SS = np.zeros(N, dtype='float64')

    S[0] = (-3.5 * f[0] + 4.0 * f[1] - 0.5 * f[2]) / dx
    SS[0] = (34.0 / 3.0 * f[0] - 83.0 / 4.0 * f[1] + 10.0 * f[2] - 7.0 / 12.0 * f[3]) / dx ** 2

    S[-1] = -(-3.5 * f[-1] + 4.0 * f[-2] - 0.5 * f[-3]) / dx
    SS[-1] = (34.0 / 3.0 * f[-1] - 83.0 / 4.0 * f[-2] + 10.0 * f[-3] - 7.0 / 12.0 * f[-4]) / dx ** 2

    for i in prange(1, N - 1):
        S[i] = 15.0 / 16.0 * (f[i + 1] - f[i - 1]) / dx
        SS[i] = (3.0 * f[i - 1] - 6.0 * f[i] + 3.0 * f[i + 1]) / dx ** 2

    return np.stack((S, SS))


@njit(float64[:](float64[:], float64[:], float64), cache=True)
def CCD(f: np.ndarray, c: np.ndarray, dx: float64):
    A, B, AA, BB = CCD_coeffs(f.size, dx)
    S, SS = CCD_src(f, f.size, dx)
    fx, fxx = twin_bks(A, B, AA, BB, S, SS)
    return fx


@njit(float64[:, :](float64[:], float64[:], float64), cache=True)
def CCD_full(f: np.ndarray, c: np.ndarray, dx: float64):
    A, B, AA, BB = CCD_coeffs(f.size, dx)
    S, SS = CCD_src(f, f.size, dx)
    fx, fxx = twin_bks(A, B, AA, BB, S, SS)
    return np.stack((fx, fxx))
