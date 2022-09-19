import cupy as cp
import numpy as np
from math import ceil
from munch import Munch
from typing import Callable


class Bridge:
    def __init__(self, data, use_cuda=False):
        self.cpu = np.copy(data)
        self.use_cuda = use_cuda
        if use_cuda:
            self.gpu = cp.array(self.cpu)
        else:
            self.gpu = None


class Scalar:

    def __init__(self, _size: list[int], ghc: int, _axis_data: list[tuple[float, float]],
                 num_of_data: int = 1, no_axis=False, no_data=False, use_cuda=False, threadsperblock=16):

        self.use_cuda = use_cuda
        self.ndim = len(_size)
        self.ghc = ghc

        ghc_array = [ghc for _ in range(self.ndim)]
        size = [x + 1 for x in _size]
        axis_data = [x for x in _axis_data]
        to_cell = [1 for _ in range(self.ndim)]
        while len(size) < 3:
            size.append(1)
            ghc_array.append(0)
            axis_data.append((0, 0))
            to_cell.append(0)

        self.shape = np.array(size, dtype=int)

        axis_data = np.array(axis_data, dtype=np.dtype('float64, float64'))
        self.x = Bridge(np.linspace(*axis_data[0], num=self.shape[0]), use_cuda)
        self.dx = self.x.cpu[1] - self.x.cpu[0]
        self.xc = Bridge((np.arange(self.x.cpu.size - 1, dtype=float) + 0.5) * self.dx, use_cuda)

        self.y = Bridge(np.linspace(*axis_data[1], num=self.shape[1] if self.shape[1] > 1 else 2), use_cuda)
        self.dy = self.y.cpu[1] - self.y.cpu[0]
        self.yc = Bridge((np.arange(self.y.cpu.size - 1, dtype=float) + 0.5) * self.dy, use_cuda)

        self.z = Bridge(np.linspace(*axis_data[2], num=self.shape[2] if self.shape[2] > 1 else 2), use_cuda)
        self.dz = self.z.cpu[1] - self.z.cpu[0]
        self.zc = Bridge((np.arange(self.z.cpu.size - 1, dtype=float) + 0.5) * self.dz, use_cuda)

        self.grids = np.array([self.dx, self.dy, self.dz], dtype='float64')

        self.h = min(self.grids[:self.ndim])
        self.dv = np.product(self.grids[:self.ndim])

        array_shape = np.array(ghc_array, dtype=int) * 2 + self.shape - np.array(to_cell)
        if not no_data:
            self.data = Bridge(
                np.stack([np.zeros(array_shape, dtype=np.float64) for _ in range(num_of_data)]), use_cuda)

        if no_axis:
            del self.x, self.y, self.z

        self.threadsperblock = (threadsperblock, threadsperblock, threadsperblock)
        self.blockspergrid = tuple([int(ceil(array_shape[i] / self.threadsperblock[i])) for i in range(3)])

    @property
    def core(self):
        if self.ndim == 1:
            return self.data.cpu[0, self.ghc:-self.ghc, 0, 0]
        elif self.ndim == 2:
            return self.data.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, 0]
        else:
            return self.data.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc]

    @core.setter
    def core(self, value):
        if self.ndim == 1:
            self.data.cpu[0, self.ghc:-self.ghc, 0, 0] = value
        elif self.ndim == 2:
            self.data.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, 0] = value
        else:
            self.data.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc] = value

    @property
    def mesh(self):
        geo = Munch()
        if self.ndim == 1:
            geo.x = self.xc.cpu
            return geo
        elif self.ndim == 2:
            x, y = np.meshgrid(self.xc.cpu, self.yc.cpu, indexing='ij')
            geo.x = x
            geo.y = y
            return geo
        else:
            x, y, z = np.meshgrid(self.xc.cpu, self.yc.cpu, self.zc.cpu, indexing='ij')
            geo.x = x
            geo.y = y
            geo.z = z
            return geo

    def to_host(self):
        if self.use_cuda:
            self.data.cpu = self.data.gpu.get()

    def to_device(self):
        if self.use_cuda:
            del self.data.gpu
            self.data.gpu = cp.array(self.data.cpu)


class Vector(Scalar):

    def __init__(self, _size: list[int], ghc: int, _axis_data: list[tuple[float, float]], num_of_data: int = 1,
                 no_axis=True, no_data=False, use_cuda=False, use_staggered=False, threadsperblock=16):

        super().__init__(_size, ghc, _axis_data, num_of_data, no_axis, no_data, use_cuda, threadsperblock)

        self.x = Munch()
        self.x.cell = Bridge(self.data.cpu, use_cuda)
        if use_staggered:
            self.x.face = Bridge(self.data.cpu, use_cuda)

        if self.ndim > 1:
            self.y = Munch()
            self.y.cell = Bridge(self.data.cpu, use_cuda)
            if use_staggered:
                self.y.face = Bridge(self.data.cpu, use_cuda)

        if self.ndim > 2:
            self.z = Munch()
            self.z.cell = Bridge(self.data.cpu, use_cuda)
            if use_staggered:
                self.z.face = Bridge(self.data.cpu, use_cuda)

        self.use_staggered = use_staggered
        del self.data

    @property
    def cell_of0(self):
        if self.ndim == 1:
            return np.stack([self.x.cell.cpu[0]])
        elif self.ndim == 2:
            return np.stack([self.x.cell.cpu[0], self.y.cell.cpu[0]])
        else:
            return np.stack([self.x.cell.cpu[0], self.y.cell.cpu[0], self.z.cell.cpu[0]])

    @cell_of0.setter
    def cell_of0(self, value):
        if self.ndim == 1:
            self.x.cell.cpu[0] = value
        elif self.ndim == 2:
            self.x.cell.cpu[0] = value[0]
            self.y.cell.cpu[0] = value[1]
        else:
            self.x.cell.cpu[0] = value[0]
            self.y.cell.cpu[0] = value[1]
            self.z.cell.cpu[0] = value[2]

    @property
    def face_of0(self):
        if self.ndim == 1:
            return np.stack([self.x.face.cpu[0]])
        elif self.ndim == 2:
            return np.stack([self.x.face.cpu[0], self.y.face.cpu[0]])
        else:
            return np.stack([self.x.face.cpu[0], self.y.face.cpu[0], self.z.face.cpu[0]])

    @face_of0.setter
    def face_of0(self, value):
        if self.ndim == 1:
            self.x.face.cpu[0] = value
        elif self.ndim == 2:
            self.x.face.cpu[0] = value[0]
            self.y.face.cpu[0] = value[1]
        else:
            self.x.face.cpu[0] = value[0]
            self.y.face.cpu[0] = value[1]
            self.z.face.cpu[0] = value[2]

    @property
    def core(self):
        if self.ndim == 1:
            return self.x.cell.cpu[0, self.ghc:-self.ghc, 0, 0]
        elif self.ndim == 2:
            x = self.x.cell.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, 0]
            y = self.y.cell.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, 0]
            return np.stack((x, y))
        else:
            x = self.x.cell.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc]
            y = self.y.cell.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc]
            z = self.z.cell.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc]
            return np.stack((x, y, z))

    @core.setter
    def core(self, value):
        if self.ndim == 1:
            self.x.cell.cpu[0, self.ghc:-self.ghc, 0, 0] = value
        elif self.ndim == 2:
            self.x.cell.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, 0] = value[0]
            self.y.cell.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, 0] = value[1]
        else:
            self.x.cell.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc] = value[0]
            self.y.cell.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc] = value[1]
            self.z.cell.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc] = value[2]

    def to_host(self):

        if self.use_cuda:

            self.x.cell.cpu = self.x.cell.gpu.get()
            if self.use_staggered:
                self.x.face.cpu = self.x.face.gpu.get()

            if self.ndim > 1:
                self.y.cell.cpu = self.y.cell.gpu.get()
                if self.use_staggered:
                    self.y.face.cpu = self.y.face.gpu.get()

            if self.ndim > 2:
                self.z.cell.cpu = self.z.cell.gpu.get()
                if self.use_staggered:
                    self.z.face.cpu = self.z.face.gpu.get()

    def to_device(self):

        if self.use_cuda:

            del self.x.cell.gpu
            self.x.cell.gpu = cp.array(self.x.cell.cpu)
            if self.use_staggered:
                del self.x.face.gpu
                self.x.face.gpu = cp.array(self.x.face.cpu)

            if self.ndim > 1:
                del self.y.cell.gpu
                self.y.cell.gpu = cp.array(self.y.cell.cpu)
                if self.use_staggered:
                    del self.y.face.gpu
                    self.y.face.gpu = cp.array(self.y.face.cpu)

            if self.ndim > 2:
                del self.z.cell.gpu
                self.z.cell.gpu = cp.array(self.z.cell.cpu)
                if self.use_staggered:
                    del self.z.face.gpu
                    self.z.face.gpu = cp.array(self.z.face.cpu)

    def apply_bc_for_cell(self, bc: Callable):

        bc(self.x.cell.cpu[0], self.ghc, self.ndim)
        if self.ndim > 1:
            bc(self.y.cell.cpu[0], self.ghc, self.ndim)
        if self.ndim > 2:
            bc(self.z.cell.cpu[0], self.ghc, self.ndim)

    def apply_bc_for_face(self, bc: Callable):

        bc(self.x.face.cpu[0], self.ghc, self.ndim)
        if self.ndim > 1:
            bc(self.y.face.cpu[0], self.ghc, self.ndim)
        if self.ndim > 2:
            bc(self.z.face.cpu[0], self.ghc, self.ndim)
