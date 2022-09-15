import cupy as cp
import numpy as np
from math import ceil
from munch import Munch


class Bridge:
    def __init__(self, data, use_cuda=False):
        self.cpu = np.copy(data)
        self.use_cuda = use_cuda
        if use_cuda:
            self.gpu = cp.asarray(self.cpu)
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
        while len(size) < 3:
            size.append(1)
            ghc_array.append(0)
            axis_data.append((0, 0))

        self.shape = np.array(size, dtype=int)

        if not no_axis:
            axis_data = np.array(axis_data, dtype=np.dtype('float64, float64'))
            self.x = Bridge(np.linspace(*axis_data[0], num=self.shape[0]), use_cuda)
            self.dx = self.x.cpu[1] - self.x.cpu[0]

            self.y = Bridge(np.linspace(*axis_data[1], num=self.shape[1] if self.shape[1] > 1 else 2), use_cuda)
            self.dy = self.y.cpu[1] - self.y.cpu[0]

            self.z = Bridge(np.linspace(*axis_data[2], num=self.shape[2] if self.shape[2] > 1 else 2), use_cuda)
            self.dz = self.z.cpu[1] - self.z.cpu[0]
            self.grids = np.array([self.dx, self.dy, self.dz], dtype='float64')

            self.h = min(self.grids[:self.ndim])

        array_shape = np.array(ghc_array, dtype=int) * 2 + self.shape
        if not no_data:
            self.data = Bridge(
                np.stack([np.zeros(array_shape, dtype=np.float64) for _ in range(num_of_data)]), use_cuda)
            self.buffer = np.zeros(array_shape, dtype=np.float64)

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
    def core3d(self):
        if self.ndim == 1:
            return self.data.cpu[0, self.ghc:-self.ghc, :, :]
        elif self.ndim == 2:
            return self.data.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, :]
        else:
            return self.data.cpu[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc]

    @property
    def mesh(self):
        geo = Munch()
        if self.ndim == 1:
            geo.x = self.x.cpu
            return geo
        elif self.ndim == 2:
            x, y = np.meshgrid(self.x.cpu, self.y.cpu, indexing='ij')
            geo.x = x
            geo.y = y
            return geo
        else:
            x, y, z = np.meshgrid(self.x.cpu, self.y.cpu, self.z.cpu, indexing='ij')
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


class Vector:

    def __init__(self, _size: list[int], ghc: int, _axis_data: list[tuple[float, float]],
                 num_of_data: int = 1, use_cuda=False, threadsperblock=16):
        self.x = Scalar(_size, ghc, _axis_data, num_of_data, no_axis=True, no_data=False, use_cuda=use_cuda,
                        threadsperblock=threadsperblock)

        self.ghc = self.x.ghc
        self.ndim = self.x.ndim
        self.shape = self.x.shape

        if self.ndim > 1:
            self.y = Scalar(_size, ghc, _axis_data, num_of_data, no_axis=True, no_data=False, use_cuda=use_cuda,
                            threadsperblock=threadsperblock)
        if self.ndim > 2:
            self.z = Scalar(_size, ghc, _axis_data, num_of_data, no_axis=True, no_data=False, use_cuda=use_cuda,
                            threadsperblock=threadsperblock)

    @property
    def of0(self):
        if self.ndim == 1:
            return np.stack([self.x.data.cpu[0]])
        elif self.ndim == 2:
            return np.stack([self.x.data.cpu[0], self.y.data.cpu[0]])
        else:
            return np.stack([self.x.data.cpu[0], self.y.data.cpu[0], self.z.data.cpu[0]])

    @of0.setter
    def of0(self, value):
        if self.ndim == 1:
            self.x.data.cpu[0] = value
        elif self.ndim == 2:
            self.x.data.cpu[0] = value[0]
            self.y.data.cpu[0] = value[1]
        else:
            self.x.data.cpu[0] = value[0]
            self.y.data.cpu[0] = value[1]
            self.z.data.cpu[0] = value[2]

    @property
    def core(self):
        if self.ndim == 1:
            return self.x.core
        elif self.ndim == 2:
            return np.stack([self.x.core, self.y.core])
        else:
            return np.stack([self.x.core, self.y.core, self.z.core])

    @core.setter
    def core(self, value):
        if self.ndim == 1:
            self.x.core = value[0]
        elif self.ndim == 2:
            self.x.core = value[0]
            self.y.core = value[1]
        else:
            self.x.core = value[0]
            self.y.core = value[1]
            self.z.core = value[2]

    def to_device(self):
        self.x.to_device()
        if self.ndim > 1:
            self.y.to_device()
        if self.ndim > 2:
            self.z.to_device()

    def to_host(self):
        self.x.to_host()
        if self.ndim > 1:
            self.y.to_host()
        if self.ndim > 2:
            self.z.to_host()
