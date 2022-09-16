import sys

sys.path.insert(0, '../../')
import numpy as np

from pySciHPC.core.boundary_conditions import zero_order
from pySciHPC.core.level_set_method import LevelSetFunction, solve_mpls, inteface_error
from pySciHPC.core.data import Scalar, Vector
from pySciHPC.core.pde_source.convection_equation import pure_convection_source
from pySciHPC.core.scheme.spatial import UCCD
from pySciHPC.core.scheme.temporal import rk3
from pySciHPC.utils.plotter import VTKPlotter
from pySciHPC.core import solve_hyperbolic

if __name__ == "__main__":

    geo_dict = dict(_size=[32, 32], ghc=3, _axis_data=[(0.0, 1.0), (0.0, 1.0)], num_of_data=1)
    geo = Scalar(**geo_dict, no_data=True)
    ls_dict = dict(interface_width=1.5 * geo.h, density_ratio=1.0)
    phi = LevelSetFunction(**geo_dict, no_axis=True, **ls_dict)
    vel = Vector(**geo_dict)
    plotter = VTKPlotter(geo, "Vortex2D")

    dt = 0.1 * geo.h
    period = 4.0

    t = 0.0

    phi.core = -np.sqrt((geo.mesh.x - 0.5) ** 2 + (geo.mesh.y - 0.75) ** 2) + 0.15
    ic = np.copy(phi.data.cpu[0])
    zero_order(phi.data.cpu[0], geo.ghc, geo.ndim)

    plotter.create()
    plotter.add_scalar(phi.core, "phi")
    plotter.add_vector(vel.core, "velocity")
    plotter.close()

    phi.snap()

    vel.x.core = np.sin(np.pi * geo.mesh.x) ** 2 * np.sin(2.0 * np.pi * geo.mesh.y) * np.cos(np.pi * t / period)
    vel.y.core = -np.sin(np.pi * geo.mesh.y) ** 2 * np.sin(2.0 * np.pi * geo.mesh.x) * np.cos(np.pi * t / period)
    zero_order(vel.x.data.cpu[0], geo.ghc, geo.ndim)
    zero_order(vel.y.data.cpu[0], geo.ghc, geo.ndim)

    cnt = 0
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

        vel.x.core = np.sin(np.pi * geo.mesh.x) ** 2 * np.sin(2.0 * np.pi * geo.mesh.y) * np.cos(np.pi * t / period)
        vel.y.core = -np.sin(np.pi * geo.mesh.y) ** 2 * np.sin(2.0 * np.pi * geo.mesh.x) * np.cos(np.pi * t / period)
        zero_order(vel.x.data.cpu[0], geo.ghc, geo.ndim)
        zero_order(vel.y.data.cpu[0], geo.ghc, geo.ndim)

        if cnt % int(0.25 / dt) == 0:
            plotter.create()
            plotter.add_scalar(phi.core, "phi")
            plotter.add_vector(vel.core, "velocity")
            plotter.close()

    plotter.joins(dt * int(0.25 / dt))

    print(inteface_error(ic, phi.data.cpu[0]))
