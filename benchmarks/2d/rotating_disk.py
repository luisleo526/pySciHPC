import numpy as np

from pySciHPC.core import solve_hyperbolic
from pySciHPC.core.boundary_conditions import zero_order
from pySciHPC.core.data import Scalar, Vector
from pySciHPC.core.level_set_method import LevelSetFunction, solve_redistance, solve_mpls
from pySciHPC.core.pde_source.convection_equation import pure_convection_source
from pySciHPC.core.scheme.spatial import UCCD
from pySciHPC.core.scheme.temporal import rk3
from pySciHPC.utils.init_func import as_density_2d
from pySciHPC.utils.plotter import VTKPlotter

if __name__ == "__main__":
    geo_dict = dict(_size=[100, 100], ghc=3, _axis_data=[(0.0, 1.0), (0.0, 1.0)], num_of_data=1)
    geo = Scalar(**geo_dict, no_data=True)
    ls_dict = dict(interface_width=1.5 * geo.h, density_ratio=1.0)
    phi = LevelSetFunction(**geo_dict, no_axis=True, **ls_dict)
    vel = Vector(**geo_dict)
    plotter = VTKPlotter(geo, "RotatingDisk")

    period = 2.0 * np.pi

    phi.core = as_density_2d(phi, geo, ["(x-0.5)**2+(y-0.75)**2 <=0.15**2",
                                        ["not (abs(x - 0.5) <= 0.025 and y < 0.75 + 0.15 / 2)"]])

    vel.x.core = -2.0 * np.pi / period * (geo.mesh.y - 0.5)
    vel.y.core = 2.0 * np.pi / period * (geo.mesh.x - 0.5)

    zero_order(phi.data.cpu[0], phi.ghc, phi.ndim)
    zero_order(vel.x.data.cpu[0], phi.ghc, phi.ndim)
    zero_order(vel.y.data.cpu[0], phi.ghc, phi.ndim)

    plotter.create()
    plotter.add_scalar(phi.core, "phi")
    plotter.close()

    solve_redistance(phi, period=4.0, cfl=0.1, init=True)

    plotter.create()
    plotter.add_scalar(phi.core, "phi")
    plotter.close()

    phi.snap()

    t = 0.0
    cnt = 0
    dt = 0.1 * geo.h

    while t < period:

        solve_hyperbolic(phi, vel, geo, rk3, zero_order, pure_convection_source, dt, UCCD)
        solve_mpls(phi)
        phi.snap()

        if cnt % 5 == 0:
            print(f"time: {t}")
            phi.print_error()
            print("=" * 30)

        t = t + dt
        cnt += 1

        if cnt % int(period / 4 / dt) == 0:
            plotter.create()
            plotter.add_scalar(phi.core, "phi")
            plotter.add_vector(vel.core, "velocity")
            plotter.close()

    plotter.joins(dt)
