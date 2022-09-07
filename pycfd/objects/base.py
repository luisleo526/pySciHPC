import numpy as np
from munch import Munch


class Scalar:

    def __init__(self, _size: list[int], ghc: int, _axis_data: list[tuple[float, float]],
                 num_of_data: int = 1, no_axis=False, no_data=False):

        assert len(_size) == len(_axis_data), f"{_size}, {_axis_data}"
        assert all([x[1] > x[0] for x in _axis_data])

        self.ndim = len(_size)
        self.ghc = ghc

        ghc_array = [ghc for _ in range(self.ndim)]
        size = [x + 1 for x in _size]
        axis_data = [x for x in _axis_data]
        while len(size) < 3:
            size += [1]
            ghc_array += [0]
            axis_data += [(0, 0)]

        self.shape = np.array(size, dtype=int)

        if not no_axis:
            axis_data = np.array(axis_data, dtype=np.dtype('float64, float64'))
            self.x = np.linspace(*axis_data[0], num=self.shape[0])
            self.dx = self.x[1] - self.x[0]

            self.y = np.linspace(*axis_data[1], num=self.shape[1] if self.shape[1] > 1 else 2)
            self.dy = self.y[1] - self.y[0]

            self.z = np.linspace(*axis_data[2], num=self.shape[2] if self.shape[2] > 1 else 2)
            self.dz = self.z[1] - self.z[0]
            self.grids = np.array([self.dx, self.dy, self.dz], dtype='float64')

        if not no_data:
            ghc_array = np.array(ghc_array, dtype=int)
            self.data = np.stack([np.zeros(ghc_array * 2 + self.shape, dtype=np.float64) for _ in range(num_of_data)])

    @property
    def core(self):
        if self.ndim == 1:
            return self.data[0, self.ghc:-self.ghc, 0, 0]
        elif self.ndim == 2:
            return self.data[0, self.ghc:-self.ghc, self.ghc:-self.ghc, 0]
        else:
            return self.data[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc]

    @core.setter
    def core(self, value):
        if self.ndim == 1:
            self.data[0, self.ghc:-self.ghc, 0, 0] = value
        elif self.ndim == 2:
            self.data[0, self.ghc:-self.ghc, self.ghc:-self.ghc, 0] = value
        else:
            self.data[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc] = value

    @property
    def core3d(self):
        if self.ndim == 1:
            return self.data[0, self.ghc:-self.ghc, :, :]
        elif self.ndim == 2:
            return self.data[0, self.ghc:-self.ghc, self.ghc:-self.ghc, :]
        else:
            return self.data[0, self.ghc:-self.ghc, self.ghc:-self.ghc, self.ghc:-self.ghc]

    @property
    def mesh(self):
        geo = Munch()
        if self.ndim == 1:
            geo.x = self.x
            return geo
        elif self.ndim == 2:
            x, y = np.meshgrid(self.x, self.y, indexing='ij')
            geo.x = x
            geo.y = y
            return geo
        else:
            x, y, z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
            geo.x = x
            geo.y = y
            geo.z = z
            return geo


class Vector:

    def __init__(self, _size: list[int], ghc: int, _axis_data: list[tuple[float, float]],
                 num_of_data: int = 1):
        self.x = Scalar(_size, ghc, _axis_data, num_of_data, True, False)
        self.y = Scalar(_size, ghc, _axis_data, num_of_data, True, False)
        self.z = Scalar(_size, ghc, _axis_data, num_of_data, True, False)

        self.ghc = self.x.ghc
        self.ndim = self.x.ndim
        self.shape = self.x.shape

        assert self.ndim > 1

    def of(self, i):
        if self.ndim == 2:
            return np.stack([self.x.data[i], self.y.data[i]])
        else:
            return np.stack([self.x.data[i], self.y.data[i], self.z.data[i]])

    @property
    def core(self):
        if self.ndim == 2:
            return np.stack([self.x.core, self.y.core])
        else:
            return np.stack([self.x.core, self.y.core, self.z.core])
