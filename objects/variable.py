import numpy as np


class Var:

    def __init__(self, size: list[int], ghc: int, axis_data: list[tuple[float, float]],
                 DataBlock=np.zeros):

        assert len(size) == len(axis_data)
        assert all([x[1] > x[0] for x in axis_data])

        self.ghc = ghc
        ghc = [ghc for _ in range(len(size))]
        while len(size) < 3:
            size += [1]
            ghc += [0]
            axis_data += [(0, 0)]

        size = np.array(size, dtype=int)
        ghc = np.array(ghc, dtype=int)
        axis_data = np.array(axis_data, dtype=np.dtype('float, float'))
        mesh = (axis_data['f1'] - axis_data['f0']) / (size - 1)

        for i in range(3):
            match i:
                case 0:
                    self.dx = mesh[i]
                    self.x_axis = np.linspace(*axis_data[i], num=size[i])
                case 1:
                    self.dy = mesh[i]
                    self.y_axis = np.linspace(*axis_data[i], num=size[i])
                case 2:
                    self.dz = mesh[i]
                    self.z_axis = np.linspace(*axis_data[i], num=size[i])

        self.data = DataBlock(ghc * 2 + size, dtype=np.float64)
        self.size = np.product(size)
        self.shape = tuple(size)
