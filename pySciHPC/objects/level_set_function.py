from .base import Scalar
from pySciHPC.functions.level_set import Heaviside
import numpy as np
from numba import float64, njit, prange, int32


@njit(float64[:](float64[:, :, :], float64, float64, float64, int32, int32), fastmath=True, parallel=True, nogil=True)
def find_mass_vol(x, eta, density, dv, ndim, ghc):
    h = Heaviside(x, eta)
    vol = 0.0
    mass = 0.0
    if ndim == 2:
        for i in prange(ghc, x.shape[0] - ghc):
            for j in prange(ghc, x.shape[1] - ghc):
                for k in prange(x.shape[2]):
                    vol += h[i, j, k] * dv
                    mass += h[i, j, k] * (h[i, j, k] + (1.0 - h[i, j, k]) * density) * dv
    else:
        for i in prange(ghc, x.shape[0] - ghc):
            for j in prange(ghc, x.shape[1] - ghc):
                for k in prange(ghc, x.shape[2] - ghc):
                    vol += h[i, j, k] * dv
                    mass += h[i, j, k] * (h[i, j, k] + (1.0 - h[i, j, k]) * density) * dv
    return np.array([vol, mass], dtype='float64')


class LevelSetFunction(Scalar):
    def __init__(self, _size: list[int], ghc: int, _axis_data: list[tuple[float, float]], num_of_data: int = 1,
                 no_axis=False, no_data=False, use_cuda=False, interface_width: float = 0.0, density_ratio: float = 1.0):
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
