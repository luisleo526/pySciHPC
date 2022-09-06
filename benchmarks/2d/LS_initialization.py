import matplotlib.pyplot as plt
import numpy as np

from boundary_conditions.zero_order import zero_order
from objects.variable import Var
from pde.redistance_eqaution import solve_redistance
from scheme.temporal.runge_kutta import rk3

if __name__ == "__main__":
    phi = Var([64, 64], 3, [(0.0, 1.0), (0.0, 1.0)])
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if phi.x[i] <= 0.35 and phi.y[j] <= 0.35:
                phi.core()[i, j] = 1.0
            elif phi.x[i] >= 0.85 and phi.y[j] >= 0.85:
                phi.core()[i, j] = 1.0
            else:
                phi.core()[i, j] = -1.0
    zero_order(phi.data[0], phi.ghc, phi.ndim)
    phi.data[0] = solve_redistance(rk3, phi.data[0], phi.grids, phi.ghc, phi.ndim, 1.5 * phi.dx, 0.5 * phi.dx, 5.0,
                                   True)
    X, Y = np.meshgrid(phi.x, phi.y)
    fig, ax = plt.subplots()
    CS = ax.contourf(X, Y, phi.core())
