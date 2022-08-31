from objects.variable import Var
import numpy as np


phi = Var([65], 3, [(-1.0, 1.0)])
for i in range(phi.size):
    phi.data[i + phi.ghc] = np.sin(np.pi * phi.x_axis[i])

c = 1.0
dt = phi.dx * 0.01
print(phi.size, phi.data.shape, phi.data.size)
