import numpy as np
from numba import int32, float64  # import the types
from numba.experimental import jitclass

spec = [
    ('ghc', int32),
    ('x_axis', float64[:]),
    ('y_axis', float64[:]),
    ('z_axis', float64[:]),
    ('dx', float64),
    ('dy', float64),
    ('dz', float64),
    ('data', float64[:, :, :]),
    ('size', int32),
    ('shape', int32[:])
]


@jitclass(spec)
class Var:

    def __init__(self, ghc, x_axis, y_axis, z_axis, data, size, shape, dx, dy, dz):
        self.ghc = ghc

        self.x_axis = x_axis
        self.dx = dx
        self.y_axis = y_axis
        self.dy = dy
        self.z_axis = z_axis
        self.dz = dz

        self.data = data
        self.size = size
        self.shape = shape


def create_Var(size: list[int], ghc: int, axis_data: list[tuple[float, float]]):
    assert len(size) == len(axis_data)
    assert all([x[1] > x[0] for x in axis_data])
    ghc_array = [ghc for _ in range(len(size))]
    while len(size) < 3:
        size += [1]
        ghc_array += [0]
        axis_data += [(0, 0)]

    size = np.array(size, dtype=int)
    ghc_array = np.array(ghc_array, dtype=int)
    axis_data = np.array(axis_data, dtype=np.dtype('float, float'))

    x_axis = np.linspace(*axis_data[0], num=size[0])
    dx = x_axis[1] - x_axis[0]

    y_axis = np.linspace(*axis_data[1], num=size[1] if size[1] > 1 else 2)
    dy = y_axis[1] - y_axis[0]

    z_axis = np.linspace(*axis_data[2], num=size[2] if size[2] > 1 else 2)
    dz = z_axis[1] - z_axis[0]

    data = np.zeros(ghc_array * 2 + size, dtype=np.float64)
    shape = size
    size = np.product(size)

    return Var(ghc, x_axis, y_axis, z_axis, data, size, shape, dx, dy, dz)
