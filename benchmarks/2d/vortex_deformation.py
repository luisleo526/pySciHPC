import sys

sys.path.insert(0, '../../pySciHPC')
import numpy as np
from numba import set_num_threads

from pySciHPC.boundary_conditions import zero_order
from pySciHPC.functions.gradients import CCD_grad
from pySciHPC.objects import LevelSetFunction
from pySciHPC.objects import Scalar, Vector
from pySciHPC.pde_source.convection_equation import pure_convection_source
from pySciHPC.pde_source.mass_preserving_level_set import mpls_source, mpls_criterion
from pySciHPC.scheme.spatial import UCCD
from pySciHPC.scheme.temporal import rk3
from pySciHPC.utils import VTKPlotter
from pySciHPC.utils import l2_norm
from pySciHPC import solve_hyperbolic, solve_hyperbolic_steady

if __name__ == "__main__":

    set_num_threads(16)

    geo_dict = dict(_size=[32, 64], ghc=3, _axis_data=[(0.0, 1.0), (0.0, 1.0)], num_of_data=1)
    geo = Scalar(**geo_dict, no_data=True)
    ls_dict = dict(interface_width=1.5 * geo.dx, density_ratio=1.0)
    phi = LevelSetFunction(**geo_dict, no_axis=True, **ls_dict)
    vel = Vector(**geo_dict)
    plotter = VTKPlotter(geo, "Vortex2D")

    dt = 0.1 * geo.dx
    period = 4.0

    t = 0.0

    phi.core = -np.sqrt((geo.mesh.x - 0.5) ** 2 + (geo.mesh.y - 0.75) ** 2) + 0.15
    ic = np.copy(phi.core)
    zero_order(phi.data.cpu[0], geo.ghc, geo.ndim)

    plotter.create()
    plotter.add_scalar(phi.core, "phi")
    plotter.add_vector(vel.core, "velocity")
    plotter.close()

    phi.snap(np.product(geo.grids[:geo.ndim]))

    vel.x.core = np.sin(np.pi * geo.mesh.x) ** 2 * np.sin(2.0 * np.pi * geo.mesh.y) * np.cos(np.pi * t / period)
    vel.y.core = -np.sin(np.pi * geo.mesh.y) ** 2 * np.sin(2.0 * np.pi * geo.mesh.x) * np.cos(np.pi * t / period)
    zero_order(vel.x.data.cpu[0], geo.ghc, geo.ndim)
    zero_order(vel.y.data.cpu[0], geo.ghc, geo.ndim)

    cnt = 0
    while t < period:

        solve_hyperbolic(phi, vel, geo, rk3, zero_order, pure_convection_source, dt, UCCD)
        solve_hyperbolic_steady(phi, vel, geo, rk3, zero_order, mpls_source, 1.0, None, mpls_criterion, 1.0e-10, 0.0,
                                CCD_grad, phi.interface_width, phi.mass[0], phi.density_ratio)

        if cnt % 5 == 0:
            print(f"time: {t}")
            phi.print_error(np.product(geo.grids[:geo.ndim]))
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

    print(l2_norm(ic, phi.core))
