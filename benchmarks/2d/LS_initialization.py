import sys

sys.path.insert(0, '../../')

from pySciHPC.core.boundary_conditions import zero_order
from pySciHPC.core.data import Scalar
from pySciHPC.core.level_set_method import LevelSetFunction, solve_redistance
from pySciHPC.utils.plotter import VTKPlotter

import matplotlib.pyplot as plt

if __name__ == "__main__":
    geo_dict = dict(_size=[64, 128], ghc=3, _axis_data=[(0.0, 1.0), (0.0, 1.0)])
    geo = Scalar(**geo_dict, no_data=True)
    ls_dict = dict(interface_width=1.5 * geo.dy, density_ratio=1.0)
    phi = LevelSetFunction(**geo_dict, no_axis=True, **ls_dict)
    plotter = VTKPlotter(geo, "LS_initialization")

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if geo.x.cpu[i] <= 0.35 and geo.y.cpu[j] <= 0.35:
                phi.core[i, j] = 1.0
            elif geo.x.cpu[i] >= 0.85 and geo.y.cpu[j] >= 0.85:
                phi.core[i, j] = 1.0
            else:
                phi.core[i, j] = -1.0
    zero_order(phi.data.cpu[0], phi.ghc, phi.ndim)

    plotter.create()
    plotter.add_scalar(phi.core, "phi")
    plotter.close()

    solve_redistance(phi, geo, period=4.0, cfl=0.1, init=True)

    plotter.create()
    plotter.add_scalar(phi.core, "phi")
    plotter.close()

    plotter.joins(1.0)
