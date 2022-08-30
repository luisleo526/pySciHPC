import numpy as np
import matplotlib.pyplot as plt


class Var1D:

    def __init__(self, size: int, ghc: int, axis_data: (float, float)):
        assert axis_data[0] < axis_data[1]
        self.axis = np.zeros(size)
        mesh = (axis_data[1] - axis_data[0]) / (size - 1)
        for i in range(size):
            self.axis[i] = axis_data[0] + (i - 1) * mesh
        self.data = np.zeros(ghc * 2 + size, dtype=np.float64)
        self.size = size
        self.ghc = ghc
        self.mesh = mesh

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def periodic_bc(self):
        for i in range(self.ghc):
            self.data[i] = self.data[self.size + i - 1]
            self.data[self.size + self.ghc + i] = self.data[self.ghc + i + 1]

    def plot(self):
        plt.plot(self.axis, self.data[self.ghc:self.size + self.ghc])


def weno_js(f: Var1D):

    fp = np.zeros_like(f.data)
    fm = np.zeros_like(f.data)
    epsilon = 1.0e-10



phi = Var1D(65, 3, (-1.0, 1.0))
for i in range(phi.size):
    phi[i + phi.ghc] = np.sin(np.pi * phi.axis[i])
phi.periodic_bc()
