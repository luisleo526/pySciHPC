import numpy as np
from numba import int32, float64, njit, prange

from ..boundary_conditions.zero_order import zero_order_x, zero_order_y, zero_order_z, zero_order
from ..functions.derivatives import find_fx, find_fy, find_fz
from ..scheme.spatial.CCD import CCD


@njit(float64(float64, float64, float64, float64))
def phyn(a, b, c, d):
    eps = 1.0e-8
    is0 = 13.0 * (a - b) ** 2.0 + 3.0 * (a - 3.0 * b) ** 2.0
    is1 = 13.0 * (b - c) ** 2.0 + 3.0 * (b + c) ** 2.0
    is2 = 13.0 * (c - d) ** 2.0 + 3.0 * (3.0 * c - d) ** 2.0
    alp0 = 1.0 / (eps + is0) ** 2.0
    alp1 = 6.0 / (eps + is1) ** 2.0
    alp2 = 3.0 / (eps + is2) ** 2.0
    w0 = alp0 / (alp0 + alp1 + alp2)
    w2 = alp2 / (alp0 + alp1 + alp2)
    return w0 / 3.0 * (a - 2.0 * b + c) + (w2 - 0.5) / 6.0 * (b - 2.0 * c + d)


@njit(float64[:, :, :](float64[:, :, :], float64[:], int32, int32), parallel=True, fastmath=True, nogil=True)
def godunov_wenojs(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32):
    assert ndim > 1
    dx, dy, dz = grids
    grad = np.zeros_like(f)

    up = np.zeros_like(f)
    um = np.zeros_like(f)

    vp = np.zeros_like(f)
    vm = np.zeros_like(f)

    wp = np.zeros_like(f)
    wm = np.zeros_like(f)

    for j in prange(f.shape[1]):
        for k in prange(f.shape[2]):
            for i in prange(ghc, f.shape[0] - ghc):
                v = 1.0 / (12.0 * dx) * (-(f[i - 1, j, k] - f[i - 2, j, k])
                                         + 7.0 * (f[i, j, k] - f[i - 1, j, k])
                                         + 7.0 * (f[i + 1, j, k] - f[i, j, k])
                                         - (f[i + 2, j, k] - f[i + 1, j, k]))

                up[i, j, k] = v + 1.0 / dx * phyn((f[i + 3, j, k] - 2.0 * f[i + 2, j, k] + f[i + 1, j, k]),
                                                  (f[i + 2, j, k] - 2.0 * f[i + 1, j, k] + f[i, j, k]),
                                                  (f[i + 1, j, k] - 2.0 * f[i, j, k] + f[i - 1, j, k]),
                                                  (f[i, j, k] - 2.0 * f[i - 1, j, k] + f[i - 2, j, k]))
                um[i, j, k] = v - 1.0 / dx * phyn((f[i - 3, j, k] - 2.0 * f[i - 2, j, k] + f[i - 1, j, k]),
                                                  (f[i - 2, j, k] - 2.0 * f[i - 1, j, k] + f[i, j, k]),
                                                  (f[i - 1, j, k] - 2.0 * f[i, j, k] + f[i + 1, j, k]),
                                                  (f[i, j, k] - 2.0 * f[i + 1, j, k] + f[i + 2, j, k]))

    zero_order_x(up, ghc)
    zero_order_x(um, ghc)

    for i in prange(f.shape[0]):
        for k in prange(f.shape[2]):
            for j in prange(ghc, f.shape[1] - ghc):
                v = 1.0 / (12.0 * dy) * (-(f[i, j - 1, k] - f[i, j - 2, k])
                                         + 7.0 * (f[i, j, k] - f[i, j - 1, k])
                                         + 7.0 * (f[i, j + 1, k] - f[i, j, k])
                                         - (f[i, j + 2, k] - f[i, j + 1, k]))

                vp[i, j, k] = v + 1.0 / dy * phyn((f[i, j + 3, k] - 2.0 * f[i, j + 2, k] + f[i, j + 1, k]),
                                                  (f[i, j + 2, k] - 2.0 * f[i, j + 1, k] + f[i, j, k]),
                                                  (f[i, j + 1, k] - 2.0 * f[i, j, k] + f[i, j - 1, k]),
                                                  (f[i, j, k] - 2.0 * f[i, j - 1, k] + f[i, j - 2, k]))
                vm[i, j, k] = v - 1.0 / dy * phyn((f[i, j - 3, k] - 2.0 * f[i, j - 2, k] + f[i, j - 1, k]),
                                                  (f[i, j - 2, k] - 2.0 * f[i, j - 1, k] + f[i, j, k]),
                                                  (f[i, j - 1, k] - 2.0 * f[i, j, k] + f[i, j + 1, k]),
                                                  (f[i, j, k] - 2.0 * f[i, j + 1, k] + f[i, j + 2, k]))
    zero_order_y(vp, ghc)
    zero_order_y(vm, ghc)

    if ndim > 2:
        for i in prange(f.shape[0]):
            for j in prange(f.shape[1]):
                for k in prange(ghc, f.shape[2] - ghc):
                    v = 1.0 / (12.0 * dz) * (-(f[i, j, k - 1] - f[i, j, k - 2])
                                             + 7.0 * (f[i, j, k] - f[i, j, k - 1])
                                             + 7.0 * (f[i, j, k + 1] - f[i, j, k])
                                             - (f[i, j, k + 2] - f[i, j, k + 1]))

                    wp[i, j, k] = v + 1.0 / dz * phyn((f[i, j, k + 3] - 2.0 * f[i, j, k + 2] + f[i, j, k + 1]),
                                                      (f[i, j, k + 2] - 2.0 * f[i, j, k + 1] + f[i, j, k]),
                                                      (f[i, j, k + 1] - 2.0 * f[i, j, k] + f[i, j, k - 1]),
                                                      (f[i, j, k] - 2.0 * f[i, j, k - 1] + f[i, j, k - 2]))
                    wm[i, j, k] = v - 1.0 / dz * phyn((f[i, j, k - 3] - 2.0 * f[i, j, k - 2] + f[i, j, k - 1]),
                                                      (f[i, j, k - 2] - 2.0 * f[i, j, k - 1] + f[i, j, k]),
                                                      (f[i, j, k - 1] - 2.0 * f[i, j, k] + f[i, j, k + 1]),
                                                      (f[i, j, k] - 2.0 * f[i, j, k + 1] + f[i, j, k + 2]))

        zero_order_z(wp, ghc)
        zero_order_z(wm, ghc)

    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
                upm = -min(up[i, j, k], 0.0)
                upp = max(up[i, j, k], 0.0)
                umm = -min(um[i, j, k], 0.0)
                ump = max(um[i, j, k], 0.0)

                vpm = -min(vp[i, j, k], 0.0)
                vpp = max(vp[i, j, k], 0.0)
                vmm = -min(vm[i, j, k], 0.0)
                vmp = max(vm[i, j, k], 0.0)

                wpm = -min(wp[i, j, k], 0.0)
                wpp = max(wp[i, j, k], 0.0)
                wmm = -min(wm[i, j, k], 0.0)
                wmp = max(wm[i, j, k], 0.0)

                if f[i, j, k] > 0.0:
                    grad[i, j, k] = np.sqrt(max(upm, ump) ** 2 + max(vpm, vmp) ** 2 + max(wpm, wmp) ** 2)
                else:
                    grad[i, j, k] = np.sqrt(max(upp, umm) ** 2 + max(vpp, vmm) ** 2 + max(wpp, wmm) ** 2)

    zero_order(grad, ghc, ndim)

    return grad


@njit(float64[:, :, :](float64[:, :, :], float64[:], int32, int32), parallel=True, fastmath=True, nogil=True)
def CCD_grad(f: np.ndarray, grids: np.ndarray, ghc: int32, ndim: int32):
    dx, dy, dz = grids
    fx = find_fx(f, dx, f, CCD)
    fy = find_fy(f, dy, f, CCD)
    fz = find_fz(f, dz, f, CCD)

    return np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
