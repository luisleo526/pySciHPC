from pycfd.boundary_conditions import zero_order
from pycfd.objects import Scalar
from pycfd.pde import solve_redistance
from pycfd.scheme.temporal import rk3
from pycfd.utils import VTKPlotter

if __name__ == "__main__":
    geo_dict = dict(_size=[64, 64], ghc=3, _axis_data=[(0.0, 1.0), (0.0, 1.0)])
    geo = Scalar(**geo_dict, no_data=True)
    phi = Scalar(**geo_dict, no_axis=True)
    plotter = VTKPlotter(geo, "LS_initialization")

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if geo.x[i] <= 0.35 and geo.y[j] <= 0.35:
                phi.core[i, j] = 1.0
            elif geo.x[i] >= 0.85 and geo.y[j] >= 0.85:
                phi.core[i, j] = 1.0
            else:
                phi.core[i, j] = -1.0
    zero_order(phi.data[0], phi.ghc, phi.ndim)

    plotter.create()
    plotter.add_scalar(phi.core, "phi")
    plotter.close()
    phi.data[0] = solve_redistance(rk3, phi.data[0], geo.grids, phi.ghc, phi.ndim, 1.5 * geo.dx, 0.5 * geo.dx, 5.0,
                                   True)
    plotter.create()
    plotter.add_scalar(phi.core, "phi")
    plotter.close()

    plotter.joins(1.0)
