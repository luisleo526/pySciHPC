import cupy as cp
import numpy as np
from typing import Tuple


class CudaDerivativesSolver:

    def __init__(self, shape: np.shape, grids: np.ndarray, dt: float, ndim: int, threadsperblock: Tuple[int],
                 blockpergrid: Tuple[int]):
        self.zero_buffer = cp.zeros(shape, dtype='float64')

        self.fx = cp.zeros(shape, dtype='float64')
        self.fxx = cp.zeros(shape, dtype='float64')

        self.dx = cp.ones(shape, dtype='float64') * grids[0]
        self.dy = cp.ones(shape, dtype='float64') * grids[1]
        self.dz = cp.ones(shape, dtype='float64') * grids[2]
        self.dt = cp.ones(shape, dtype='float64') * dt

        self.block_dim = threadsperblock
        self.grid_dim = blockpergrid
        self.ndim = ndim

        self.src_buffer0 = cp.zeros(shape, dtype='float64')
        self.src_buffer1 = cp.zeros(shape, dtype='float64')
        self.src_buffer2 = cp.zeros(shape, dtype='float64')

        self.sol_buffer0 = cp.zeros(shape, dtype='float64')

    def find_fx(self, f: cp.ndarray, c: cp.ndarray):
        return NotImplemented

    def find_fy(self, f: cp.ndarray, c: cp.ndarray):
        return NotImplemented

    def find_fz(self, f: cp.ndarray, c: cp.ndarray):
        return NotImplemented
