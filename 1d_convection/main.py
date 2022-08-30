from objects.variable_1d import Var1D
import numpy as np
import matplotlib.pyplot as plt


def weno_js(f: Var1D):
    fp = np.zeros_like(f.data)
    fm = np.zeros_like(f.data)
    epsilon = 1.0e-10


phi = Var1D(65, 3, (-1.0, 1.0))
for i in range(phi.size):
    phi[i + phi.ghc] = np.sin(np.pi * phi.axis[i])
phi.periodic_bc()
print(phi.axis)
