from .basic import find_mass_vol, Heaviside, Delta, Sign
from pySciHPC.core.data import Scalar


class LevelSetFunction(Scalar):
    def __init__(self, _size: list[int], ghc: int, _axis_data: list[tuple[float, float]], num_of_data: int = 1,
                 no_axis=False, no_data=False, use_cuda=False, interface_width: float = 0.0,
                 density_ratio: float = 1.0):
        super().__init__(_size, ghc, _axis_data, num_of_data, no_axis, no_data, use_cuda)

        self.interface_width = interface_width
        self.density_ratio = density_ratio
        self.vol_history = []
        self.mass_history = []

    def snap(self):
        vol, mass = find_mass_vol(self.data.cpu[0], self.interface_width, self.density_ratio, self.dv, self.ndim,
                                  self.ghc)
        self.vol_history.append(vol)
        self.mass_history.append(mass)

    @property
    def mass(self):
        return find_mass_vol(self.data.cpu[0], self.interface_width,
                             self.density_ratio, self.dv, self.ndim, self.ghc)[1]

    @property
    def vol(self):
        return find_mass_vol(self.data.cpu[0], self.interface_width,
                             self.density_ratio, self.dv, self.ndim, self.ghc)[0]

    @property
    def heaviside(self):
        return Heaviside(self.data.cpu[0], self.interface_width)

    @property
    def delta(self):
        return Delta(self.data.cpu[0], self.interface_width)

    @property
    def sign(self):
        return Sign(self.data.cpu[0], self.interface_width)

    def print_error(self, snap=False):
        if snap:
            self.snap()
        print(f"Mass error: {(1.0 - self.mass_history[-1] / self.mass_history[0]) * 100:.6E}%")
        print(f"Volume error: {(1.0 - self.vol_history[-1] / self.vol_history[0]) * 100:.6E}%")
