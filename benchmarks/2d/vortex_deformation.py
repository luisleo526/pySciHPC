import numpy as np

from pycfd.boundary_conditions import zero_order
from pycfd.objects import Scalar, Vector
from pycfd.objects import LevelSetFunction
from pycfd.pde.convection_equation import pure_convection
from pycfd.pde.mass_preserving_level_set import mpls
from pycfd.scheme.spatial import UCCD
from pycfd.scheme.temporal import rk3, euler
from pycfd.utils import VTKPlotter
from pycfd.functions.gradients import CCD_grad

if __name__ == "__main__":
    geo_dict = dict(_size=[64, 64], ghc=3, _axis_data=[(0.0, 1.0), (0.0, 1.0)], num_of_data=1)
    geo = Scalar(**geo_dict, no_data=True)
    ls_dict = dict(interface_width=1.5 * geo.dx, density_ratio=1.0)
    phi = LevelSetFunction(**geo_dict, no_axis=True, **ls_dict)
    vel = Vector(**geo_dict)
    plotter = VTKPlotter(geo, "Vortex2D")

    dt = 0.1 * geo.dx
    period = 4.0

    t = 0.0

    phi.core = -np.sqrt((geo.mesh.x - 0.5) ** 2 + (geo.mesh.y - 0.75) ** 2) + 0.15
    zero_order(phi.data[0], geo.ghc, geo.ndim)
    phi.snap(np.product(geo.grids[:geo.ndim]))

    vel.x.core = np.sin(np.pi * geo.mesh.x) ** 2 * np.sin(2.0 * np.pi * geo.mesh.y) * np.cos(np.pi * t / period)
    vel.y.core = -np.sin(np.pi * geo.mesh.y) ** 2 * np.sin(2.0 * np.pi * geo.mesh.x) * np.cos(np.pi * t / period)
    zero_order(vel.x.data[0], geo.ghc, geo.ndim)
    zero_order(vel.y.data[0], geo.ghc, geo.ndim)

    cnt = 0
    while t < period:

        phi.data[0] = pure_convection(rk3, UCCD, phi.data[0], geo.grids, geo.ghc, geo.ndim, vel.of(0), dt)
        phi.data[0] = mpls(rk3, CCD_grad, phi.data[0], geo.grids, geo.ghc, geo.ndim, phi.interface_width, phi.mass[0],
                           phi.density_ratio)

        if cnt % 5 == 0:
            print(f"time: {t}")
            phi.print_error(np.product(geo.grids[:geo.ndim]))

        t = t + dt
        cnt += 1

        vel.x.core = np.sin(np.pi * geo.mesh.x) ** 2 * np.sin(2.0 * np.pi * geo.mesh.y) * np.cos(np.pi * t / period)
        vel.y.core = -np.sin(np.pi * geo.mesh.y) ** 2 * np.sin(2.0 * np.pi * geo.mesh.x) * np.cos(np.pi * t / period)
        zero_order(vel.x.data[0], geo.ghc, geo.ndim)
        zero_order(vel.y.data[0], geo.ghc, geo.ndim)

        if cnt % 200 == 0:
            plotter.create()
            plotter.add_scalar(phi.core, "phi")
            plotter.add_vector(vel.core, "velocity")
            plotter.close()

    plotter.joins(dt * 200)
