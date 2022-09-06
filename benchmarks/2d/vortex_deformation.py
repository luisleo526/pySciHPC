import numpy as np

from objects.variable import Var
from pde.convection_equation import solve_convection
from scheme.temporal.runge_kutta import rk3

if __name__ == "__main__":
    phi = Var([64, 64], 3, [(0.0, 1.0), (0.0, 1.0)])
    vel = Var([64, 64], 3, [(0.0, 1.0), (0.0, 1.0)], 2)

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            phi.core()[i, j] = -np.sqrt((phi.x[i] - 0.5) ** 2 + (phi.y[j] - 0.75) ** 2) + 0.15
