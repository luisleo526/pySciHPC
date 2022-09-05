import numpy as np
from numba import int32, float64, njit, prange

from boundary_conditions.interpolation import zero_order_x, zero_order_y, zero_order_z
from scheme.spatial.ENO import WENO_p, WENO_m, WENO_weights_JS


@njit(float64[:, :, :](), parallel=True, fastmath=True)
def Godunov_WENO_grad(f: np.ndarray, ghc: int32, ndim: int32, dx: float64, dy: float64, dz: float64):
    grad = np.zeros_like(f)

    up = np.zeros_like(f)
    um = np.zeros_like(f)

    vp = np.zeros_like(f)
    vm = np.zeros_like(f)

    wp = np.zeros_like(f)
    wm = np.zeros_like(f)

    df = np.zeros_like(f)
    for j in prange(f.shape[1]):
        for k in prange(f.shape[2]):
            for i in prange(1, f.shape[0]):
                df[i, j, k] = (f[i, j, k] - f[i - 1, j, k]) / dx
    zero_order_x(df, ghc)
    for j in prange(f.shape[1]):
        for k in prange(f.shape[2]):
            for i in prange(ghc, f.shape[0] - ghc):
                up[i, j, k] = WENO_p(df[i - 1, j, k], df[i, j, k], df[i + 1, j, k], df[i + 2, j, k], df[i + 3, j, k],
                                     WENO_weights_JS)
                um[i, j, k] = WENO_m(df[i - 2, j, k], df[i - 1, j, k], df[i, j, k], df[i + 1, j, k], df[i + 2, j, k],
                                     WENO_weights_JS)
    zero_order_x(up, ghc)
    zero_order_x(um, ghc)

    if ndim > 1:
        for i in prange(f.shape[0]):
            for k in prange(f.shape[2]):
                for j in prange(1, f.shape[1]):
                    df[i, j, k] = (f[i, j, k] - f[i, j - 1, k]) / dy
        zero_order_y(df, ghc)
        for i in prange(f.shape[0]):
            for k in prange(f.shape[2]):
                for j in prange(ghc, f.shape[1] - ghc):
                    vp[i, j, k] = WENO_p(df[i, j - 1, k], df[i, j, k], df[i, j + 1, k], df[i, j + 2, k],
                                         df[i, j + 3, k], WENO_weights_JS)
                    vm[i, j, k] = WENO_m(df[i, j - 2, k], df[i, j - 1, k], df[i, j, k], df[i, j + 1, k],
                                         df[i, j + 2, k], WENO_weights_JS)
        zero_order_y(vp, ghc)
        zero_order_y(vm, ghc)

    if ndim > 2:
        for i in prange(f.shape[0]):
            for j in prange(f.shape[1]):
                for k in prange(f.shape[2]):
                    df[i, j, k] = (f[i, j, k] - f[i, j, k - 1]) / dz
        zero_order_z(df, ghc)
        for i in prange(f.shape[0]):
            for j in prange(f.shape[1]):
                for k in prange(ghc, f.shape[2] - ghc):
                    wp[i, j, k] = WENO_p(df[i, j, k - 1], df[i, j, k], df[i, j, k + 1], df[i, j, k + 2],
                                         df[i, j, k + 3], WENO_weights_JS)
                    wm[i, j, k] = WENO_m(df[i, j, k - 2], df[i, j, k - 1], df[i, j, k], df[i, j, k + 1],
                                         df[i, j, k + 2], WENO_weights_JS)

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

                if f.data[i, j, k] > 0.0:
                    grad[i, j, k] = np.sqrt(max(upm, ump) ** 2 + max(vpm, vmp) ** 2 + max(wpm, wmp) ** 2)
                else:
                    grad[i, j, k] = np.sqrt(max(upp, umm) ** 2 + max(vpp, vmm) ** 2 + max(wpp, wmm) ** 2)

    zero_order_x(grad, ghc)
    if ndim > 1:
        zero_order_y(grad, ghc)
    if ndim > 2:
        zero_order_z(grad, ghc)

    return grad
