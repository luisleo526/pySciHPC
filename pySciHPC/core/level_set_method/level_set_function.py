from .basic import find_mass_vol
from ..data import Scalar


class LevelSetFunction(Scalar):
    def __init__(self, _size: list[int], ghc: int, _axis_data: list[tuple[float, float]], num_of_data: int = 1,
                 no_axis=False, no_data=False, use_cuda=False, interface_width: float = 0.0,
                 density_ratio: float = 1.0):
        super().__init__(_size, ghc, _axis_data, num_of_data, no_axis, no_data, use_cuda)

        self.interface_width = interface_width
        self.density_ratio = density_ratio
        self.vol = []
        self.mass = []

    def snap(self, dv):
        tmp = find_mass_vol(self.data.cpu[0], self.interface_width, self.density_ratio, dv, self.ndim, self.ghc)
        self.vol.append(tmp[0])
        self.mass.append(tmp[1])

    def print_error(self, dv=None):
        if dv is not None:
            self.snap(dv)
        print(f"Mass error: {(1.0 - self.mass[-1] / self.mass[0]) * 100:.6E}%")
        print(f"Volume error: {(1.0 - self.vol[-1] / self.vol[0]) * 100:.6E}%")
