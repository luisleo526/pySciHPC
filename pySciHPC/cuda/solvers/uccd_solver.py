from typing import Tuple

import cupy as cp
import numpy as np
from numba import cuda
from scipy.sparse import bmat, diags
from scipy.sparse.linalg import inv

from .derivatives_solver import CudaDerivativesSolver
from ...core.scheme.spatial.UCCD import UCCD_coeffs


def make_inverse(n: int, dx: float):
    au, bu, aau, bbu, ad, bd, aad, bbd = UCCD_coeffs(n, dx)

    au = diags([au[0, 1:], au[1, :], au[2, :-1]], [-1, 0, 1], format='csc')
    bu = diags([bu[0, 1:], bu[1, :], bu[2, :-1]], [-1, 0, 1], format='csc')
    aau = diags([aau[0, 1:], aau[1, :], aau[2, :-1]], [-1, 0, 1], format='csc')
    bbu = diags([bbu[0, 1:], bbu[1, :], bbu[2, :-1]], [-1, 0, 1], format='csc')

    ad = diags([ad[0, 1:], ad[1, :], ad[2, :-1]], [-1, 0, 1], format='csc')
    bd = diags([bd[0, 1:], bd[1, :], bd[2, :-1]], [-1, 0, 1], format='csc')
    aad = diags([aad[0, 1:], aad[1, :], aad[2, :-1]], [-1, 0, 1], format='csc')
    bbd = diags([bbd[0, 1:], bbd[1, :], bbd[2, :-1]], [-1, 0, 1], format='csc')

    mu = cp.array(inv(bmat([[au, bu], [aau, bbu]], format='csc')).toarray(), dtype=float)
    md = cp.array(inv(bmat([[ad, bd], [aad, bbd]], format='csc')).toarray(), dtype=float)

    return mu, md


def src_bc(f, dx, s, ss):
    s[0] = (-3.5 * f[0] + 4.0 * f[1] - 0.5 * f[2]) / dx
    ss[0] = (34.0 / 3.0 * f[0] - 83.0 / 4.0 * f[1] + 10.0 * f[2] - 7.0 / 12.0 * f[3]) / dx ** 2

    s[-1] = -(-3.5 * f[-1] + 4.0 * f[-2] - 0.5 * f[-3]) / dx
    ss[-1] = (34.0 / 3.0 * f[-1] - 83.0 / 4.0 * f[-2] + 10.0 * f[-3] - 7.0 / 12.0 * f[-4]) / dx ** 2


@cuda.jit
def assign_src_upwind(f, dx, s, ss):
    c3 = -0.06096119008109
    c2 = 1.99692238016218
    c1 = -1.93596119008109

    i = cuda.grid(1)

    if i > 0 and i < f.size:
        s[i] = (c1 * f[i - 1] + c2 * f[i] + c3 * f[i + 1]) / dx[i]
        ss[i] = (3.0 * f[i - 1] - 6.0 * f[i] + 3.0 * f[i + 1]) / dx[i] ** 2


@cuda.jit
def assign_src_downwind(f, dx, s, ss):
    c3 = -0.06096119008109
    c2 = 1.99692238016218
    c1 = -1.93596119008109

    i = cuda.grid(1)

    if i > 0 and i < f.size:
        s[i] = -(c3 * f[i - 1] + c2 * f[i] + c1 * f[i + 1]) / dx[i]
        ss[i] = (3.0 * f[i - 1] - 6.0 * f[i] + 3.0 * f[i + 1]) / dx[i] ** 2


@cuda.jit
def choose_upwind(fx, fxd, c):
    i = cuda.grid(1)

    if i < fx.size:
        if c[i] < 0.0:
            fx[i] = fxd[i]


class CudaUCCDSovler(CudaDerivativesSolver):

    def __init__(self, shape: np.shape, grids: np.ndarray, dt: float, ndim: int, threadsperblock: Tuple[int],
                 blockpergrid: Tuple[int]):
        super().__init__(shape, grids, dt, ndim, threadsperblock, blockpergrid)

        self.buffer = cp.zeros(shape, dtype='float64')
        self.grids = grids

        self.Ax = make_inverse(shape[0], grids[0])
        self.sx = cp.zeros(shape[0], dtype=float)
        self.ssx = cp.zeros(shape[0], dtype=float)
        self.bx = cp.zeros(shape[0] * 2, dtype=float)
        self.sol_x = cp.zeros(shape[0] * 2, dtype=float)
        self.indices_x = cp.arange(shape[0], dtype=int)

        if self.ndim > 1:
            self.Ay = make_inverse(shape[1], grids[1])
            self.sy = cp.zeros(shape[1], dtype=float)
            self.ssy = cp.zeros(shape[1], dtype=float)
            self.by = cp.zeros(shape[1] * 2, dtype=float)
            self.sol_y = cp.zeros(shape[1] * 2, dtype=float)
            self.indices_y = cp.arange(shape[1], dtype=int)
        if self.ndim > 2:
            self.Az = make_inverse(shape[2], grids[2])
            self.sz = cp.zeros(shape[2], dtype=float)
            self.ssz = cp.zeros(shape[2], dtype=float)
            self.bz = cp.zeros(shape[2] * 2, dtype=float)
            self.sol_z = cp.zeros(shape[2] * 2, dtype=float)
            self.indices_z = cp.arange(shape[2], dtype=int)

    def find_fx(self, f: cp.ndarray, c: cp.ndarray):
        for j in range(f.shape[1]):
            for k in range(f.shape[2]):
                assign_src_upwind[(self.grid_dim[0]), (self.block_dim[0])](
                    f[:, j, k], self.dx[:, j, k], self.sx,self.ssx)
                src_bc(f[:, j, k], self.grids[0], self.sx, self.ssx)
                cp.concatenate((self.sx, self.ssx), out=self.bx)
                cp.dot(self.Ax[0], self.bx, out=self.sol_x)
                cp.take(self.sol_x, indices=self.indices_x, out=self.fx[:, j, k])

                assign_src_downwind[(self.grid_dim[0]), (self.block_dim[0])](
                    f[:, j, k], self.dx[:, j, k], self.sx,self.ssx)
                src_bc(f[:, j, k], self.grids[0], self.sx, self.ssx)
                cp.concatenate((self.sx, self.ssx), out=self.bx)
                cp.dot(self.Ax[1], self.bx, out=self.sol_x)
                cp.take(self.sol_x, indices=self.indices_x, out=self.buffer[:, j, k])

                choose_upwind[(self.grid_dim[0]), (self.block_dim[0])](
                    self.fx[:, j, k], self.buffer[:, j, k], c[:, j, k])
        return self.fx

    def find_fy(self, f: cp.ndarray, c: cp.ndarray):
        if self.ndim > 1:
            for i in range(f.shape[0]):
                for k in range(f.shape[2]):
                    assign_src_upwind[(self.grid_dim[1]), (self.block_dim[1])](
                        f[i, :, k], self.dy[i, :, k], self.sy, self.ssy)
                    src_bc(f[i, :, k], self.grids[1], self.sy, self.ssy)
                    cp.concatenate((self.sy, self.ssy), out=self.by)
                    cp.dot(self.Ay[0], self.by, out=self.sol_y)
                    cp.take(self.sol_y, indices=self.indices_y, out=self.fx[i, :, k])

                    assign_src_downwind[(self.grid_dim[1]), (self.block_dim[1])](
                        f[i, :, k], self.dy[i, :, k], self.sy, self.ssy)
                    src_bc(f[i, :, k], self.grids[1], self.sy, self.ssy)
                    cp.concatenate((self.sy, self.ssy), out=self.by)
                    cp.dot(self.Ay[1], self.by, out=self.sol_y)
                    cp.take(self.sol_y, indices=self.indices_y, out=self.buffer[i, :, k])

                    choose_upwind[(self.grid_dim[1]), (self.block_dim[1])](
                        self.fx[i, :, k], self.buffer[i, :, k], c[i, :, k])
            return self.fx
        else:
            return self.zero_buffer

    def find_fz(self, f: cp.ndarray, c: cp.ndarray):
        if self.ndim > 2:
            for i in range(f.shape[0]):
                for j in range(f.shape[1]):
                    assign_src_upwind[(self.grid_dim[2]), (self.block_dim[2])](
                        f[i, j, :], self.dz[i, j, :], self.sz, self.ssz)
                    src_bc(f[i, j, :], self.grids[2], self.sz, self.ssz)
                    cp.concatenate((self.sz, self.ssz), out=self.bz)
                    cp.dot(self.Az[0], self.bz, out=self.sol_z)
                    cp.take(self.sol_z, indices=self.indices_z, out=self.fx[i, j, :])

                    assign_src_upwind[(self.grid_dim[2]), (self.block_dim[2])](
                        f[i, j, :], self.dz[i, j, :], self.sz, self.ssz)
                    src_bc(f[i, j, :], self.grids[2], self.sz, self.ssz)
                    cp.concatenate((self.sz, self.ssz), out=self.bz)
                    cp.dot(self.Az[1], self.bz, out=self.sol_z)
                    cp.take(self.sol_z, indices=self.indices_z, out=self.buffer[i, j, :])

                    choose_upwind[(self.grid_dim[2]), (self.block_dim[2])](
                        self.fx[i, j, :], self.buffer[i, j, :], c[i, j, :])
            return self.fx
        else:
            return self.zero_buffer
