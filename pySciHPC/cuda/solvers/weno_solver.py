from typing import Tuple

import cupy as cp
import numpy as np
from numba import cuda

from .derivatives_solver import CudaDerivativesSolver
from ..kernels import assign_zero
from ...core.scheme.spatial.ENO import epsilon


@cuda.jit(device=True)
def js_weights(b1, b2, b3):
    a1 = 1.0 / (epsilon + b1) ** 2
    a2 = 6.0 / (epsilon + b2) ** 2
    a3 = 3.0 / (epsilon + b3) ** 2

    w1 = a1 / (a1 + a2 + a3)
    w2 = a2 / (a1 + a2 + a3)
    w3 = a3 / (a1 + a2 + a3)

    return w1, w2, w3


@cuda.jit(device=True)
def z_weights(b1, b2, b3):
    a1 = 1.0 * (1.0 + abs(b1 - b3) / (epsilon + b1))
    a2 = 6.0 * (1.0 + abs(b1 - b3) / (epsilon + b2))
    a3 = 3.0 * (1.0 + abs(b1 - b3) / (epsilon + b3))

    w1 = a1 / (a1 + a2 + a3)
    w2 = a2 / (a1 + a2 + a3)
    w3 = a3 / (a1 + a2 + a3)

    return w1, w2, w3


@cuda.jit(device=True)
def indicators_m(a, b, c, d, e):
    b1 = 13.0 * (a - 2.0 * b + c) ** 2 + 3.0 * (a - 4.0 * b + 3.0 * c) ** 2
    b2 = 13.0 * (b - 2.0 * c + d) ** 2 + 3.0 * (b - d) ** 2
    b3 = 13.0 * (c - 2.0 * d + e) ** 2 + 3.0 * (3.0 * c - 4.0 * d + e) ** 2

    return b1, b2, b3


@cuda.jit(device=True)
def indicators_p(a, b, c, d, e):
    b3 = 13.0 * (a - 2.0 * b + c) ** 2 + 3.0 * (a - 4.0 * b + 3.0 * c) ** 2
    b2 = 13.0 * (b - 2.0 * c + d) ** 2 + 3.0 * (b - d) ** 2
    b1 = 13.0 * (c - 2.0 * d + e) ** 2 + 3.0 * (3.0 * c - 4.0 * d + e) ** 2

    return b1, b2, b3


@cuda.jit
def weno_js_x(f, fp, fm):
    i, j, k = cuda.grid(3)

    if i < f.shape[0] - 3 and i > 1 and j < f.shape[1] and k < f.shape[2]:
        # Left-biased
        a = f[i - 1, j, k]
        b = f[i, j, k]
        c = f[i + 1, j, k]
        d = f[i + 2, j, k]
        e = f[i + 3, j, k]

        b1, b2, b3 = indicators_p(a, b, c, d, e)
        w1, w2, w3 = js_weights(b1, b2, b3)

        f3 = (- a + 5.0 * b + 2.0 * c) / 6.0
        f2 = (2.0 * b + 5.0 * c - d) / 6.0
        f1 = (11.0 * c - 7.0 * d + 2.0 * e) / 6.0

        fp[i, j, k] = w1 * f1 + w2 * f2 + w3 * f3

        # Right-biased
        a = f[i - 2, j, k]
        b = f[i - 1, j, k]
        c = f[i, j, k]
        d = f[i + 1, j, k]
        e = f[i + 2, j, k]

        b1, b2, b3 = indicators_m(a, b, c, d, e)
        w1, w2, w3 = js_weights(b1, b2, b3)

        f3 = (- e + 5.0 * d + 2.0 * c) / 6.0
        f2 = (2.0 * d + 5.0 * c - b) / 6.0
        f1 = (11.0 * c - 7.0 * b + 2.0 * a) / 6.0

        fm[i, j, k] = w1 * f1 + w2 * f2 + w3 * f3


@cuda.jit
def weno_js_y(f, fp, fm):
    i, j, k = cuda.grid(3)

    if j < f.shape[1] - 3 and j > 1 and i < f.shape[0] and k < f.shape[2]:
        # Left-biased
        a = f[i, j - 1, k]
        b = f[i, j, k]
        c = f[i, j + 1, k]
        d = f[i, j + 2, k]
        e = f[i, j + 3, k]

        b1, b2, b3 = indicators_p(a, b, c, d, e)
        w1, w2, w3 = js_weights(b1, b2, b3)

        f3 = (- a + 5.0 * b + 2.0 * c) / 6.0
        f2 = (2.0 * b + 5.0 * c - d) / 6.0
        f1 = (11.0 * c - 7.0 * d + 2.0 * e) / 6.0

        fp[i, j, k] = w1 * f1 + w2 * f2 + w3 * f3

        # Right-biased
        a = f[i, j - 2, k]
        b = f[i, j - 1, k]
        c = f[i, j, k]
        d = f[i, j + 1, k]
        e = f[i, j + 2, k]

        b1, b2, b3 = indicators_m(a, b, c, d, e)
        w1, w2, w3 = js_weights(b1, b2, b3)

        f3 = (- e + 5.0 * d + 2.0 * c) / 6.0
        f2 = (2.0 * d + 5.0 * c - b) / 6.0
        f1 = (11.0 * c - 7.0 * b + 2.0 * a) / 6.0

        fm[i, j, k] = w1 * f1 + w2 * f2 + w3 * f3


@cuda.jit
def weno_js_z(f, fp, fm):
    i, j, k = cuda.grid(3)

    if k < f.shape[1] - 3 and k > 1 and j < f.shape[1] and i < f.shape[0]:
        # Left-biased
        a = f[i, j, k - 1]
        b = f[i, j, k]
        c = f[i, j, k + 1]
        d = f[i, j, k + 2]
        e = f[i, j, k + 3]

        b1, b2, b3 = indicators_p(a, b, c, d, e)
        w1, w2, w3 = js_weights(b1, b2, b3)

        f3 = (- a + 5.0 * b + 2.0 * c) / 6.0
        f2 = (2.0 * b + 5.0 * c - d) / 6.0
        f1 = (11.0 * c - 7.0 * d + 2.0 * e) / 6.0

        fp[i, j, k] = w1 * f1 + w2 * f2 + w3 * f3

        # Right-biased
        a = f[i, j, k - 2]
        b = f[i, j, k - 1]
        c = f[i, j, k]
        d = f[i, j, k + 1]
        e = f[i, j, k + 2]

        b1, b2, b3 = indicators_m(a, b, c, d, e)
        w1, w2, w3 = js_weights(b1, b2, b3)

        f3 = (- e + 5.0 * d + 2.0 * c) / 6.0
        f2 = (2.0 * d + 5.0 * c - b) / 6.0
        f1 = (11.0 * c - 7.0 * b + 2.0 * a) / 6.0

        fm[i, j, k] = w1 * f1 + w2 * f2 + w3 * f3


@cuda.jit
def find_flux(fh, fp, fm, c):
    i, j, k = cuda.grid(3)
    if i < fh.shape[0] and j < fh.shape[1] and k < fh.shape[2]:
        if c[i, j, k] > 0.0:
            fh[i, j, k] = fm[i, j, k]
        else:
            fh[i, j, k] = fp[i, j, k]


@cuda.jit
def fx_from_flux(fh, fx, dx):
    i, j, k = cuda.grid(3)
    if i < fh.shape[0] and j < fh.shape[1] and k < fh.shape[2] and i > 0:
        fx[i, j, k] = (fh[i, j, k] - fh[i - 1, j, k]) / dx[i, j, k]


@cuda.jit
def fy_from_flux(fh, fx, dx):
    i, j, k = cuda.grid(3)
    if i < fh.shape[0] and j < fh.shape[1] and k < fh.shape[2] and j > 0:
        fx[i, j, k] = (fh[i, j, k] - fh[i, j - 1, k]) / dx[i, j, k]


@cuda.jit
def fz_from_flux(fh, fx, dx):
    i, j, k = cuda.grid(3)
    if i < fh.shape[0] and j < fh.shape[1] and k < fh.shape[2] and k > 0:
        fx[i, j, k] = (fh[i, j, k] - fh[i, j, k - 1]) / dx[i, j, k]


class CudaWENOSolver(CudaDerivativesSolver):

    def __init__(self, shape: np.shape, grids: np.ndarray, dt: float, ndim: int, threadsperblock: Tuple[int],
                 blockpergrid: Tuple[int]):

        super().__init__(shape, grids, dt, ndim, threadsperblock, blockpergrid)
        self.fh = cp.zeros_like(self.zero_buffer, dtype=float)
        self.fp = cp.zeros_like(self.zero_buffer, dtype=float)
        self.fm = cp.zeros_like(self.zero_buffer, dtype=float)
        self.gp = cp.zeros_like(self.zero_buffer, dtype=float)
        self.gm = cp.zeros_like(self.zero_buffer, dtype=float)
        self.hp = cp.zeros_like(self.zero_buffer, dtype=float)
        self.hm = cp.zeros_like(self.zero_buffer, dtype=float)

    def find_fx(self, f: cp.ndarray, c: cp.ndarray):
        weno_js_x[self.grid_dim, self.block_dim](f, self.fp, self.fm)
        find_flux[self.grid_dim, self.block_dim](self.fh, self.fp, self.fm, c)
        fx_from_flux[self.grid_dim, self.block_dim](self.fh, self.fx, self.dx)
        return self.fx

    def find_fy(self, f: cp.ndarray, c: cp.ndarray):
        if self.ndim > 1:
            weno_js_y[self.grid_dim, self.block_dim](f, self.gp, self.gm)
            find_flux[self.grid_dim, self.block_dim](self.fh, self.fp, self.fm, c)
            fy_from_flux[self.grid_dim, self.block_dim](self.fh, self.fx, self.dy)
            return self.fx
        else:
            return self.zero_buffer

    def find_fz(self, f: cp.ndarray, c: cp.ndarray):
        if self.ndim > 2:
            weno_js_z[self.grid_dim, self.block_dim](f, self.hp, self.hm)
            find_flux[self.grid_dim, self.block_dim](self.fh, self.fp, self.fm, c)
            fz_from_flux[self.grid_dim, self.block_dim](self.fh, self.fx, self.dz)
            return self.fx
        else:
            return self.zero_buffer

    def find_cellface(self, f: cp.ndarray):
        weno_js_x[self.grid_dim, self.block_dim](f, self.fp, self.fm)
        if self.ndim > 1:
            weno_js_y[self.grid_dim, self.block_dim](f, self.gp, self.gm)
        else:
            assign_zero(self.gp)
            assign_zero(self.gm)
        if self.ndim > 2:
            weno_js_z[self.grid_dim, self.block_dim](f, self.hp, self.hm)
        else:
            assign_zero(self.hp)
            assign_zero(self.hm)
