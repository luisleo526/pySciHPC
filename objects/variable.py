import numpy as np


class Var:

    def __init__(self, size: list[int], ghc: int, axis_data: list[tuple[float, float]]):

        assert len(size) == len(axis_data)
        assert all([x[1] > x[0] for x in axis_data])
        self.ndim = len(size)
        self.ghc = ghc
        ghc_array = [ghc for _ in range(self.ndim)]
        while len(size) < 3:
            size += [1]
            ghc_array += [0]
            axis_data += [(0, 0)]
        self.shape = np.array(size, dtype=int)
        self.size = np.product(self.shape)
        ghc_array = np.array(ghc_array, dtype=int)
        axis_data = np.array(axis_data, dtype=np.dtype('float64, float64'))

        self.x = np.linspace(*axis_data[0], num=self.shape[0])
        self.dx = self.x[1] - self.x[0]

        self.y = np.linspace(*axis_data[1], num=self.shape[1] if self.shape[1] > 1 else 2)
        self.dy = self.y[1] - self.y[0]

        self.z = np.linspace(*axis_data[2], num=self.shape[2] if self.shape[2] > 1 else 2)
        self.dz = self.z[1] - self.z[0]
        self.data = np.zeros(ghc_array * 2 + self.shape, dtype=np.float64)
