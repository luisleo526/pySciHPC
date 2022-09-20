import numpy as np
from numba import njit, float64, int32

from .derivatives import ccd_x, ccd_y, ccd_z
from ..functions.cell_face_interpolation import face_to_cell
from ..boundary_conditions import zero_order


@njit(float64[:, :, :, :](float64[:, :, :, :], float64[:], int32, int32, float64[:, :, :]), fastmath=True,
      parallel=True,
      nogil=True)
def CCD_stress(vel: np.ndarray, grids: np.ndarray, ghc: int, ndim: int, mu: np.ndarray):
    cell_vell = face_to_cell(vel, ndim)

    dx, dy, dz = grids

    mu_x, mu_xx = ccd_x(mu, dx)
    mu_y, mu_yy = ccd_y(mu, dy)
    mu_z, mu_zz = ccd_z(mu, dz)

    zero_order(vel[0], ghc, ndim)
    u_x, u_xx = ccd_x(cell_vell[0], dx)
    u_y, u_yy = ccd_y(cell_vell[0], dy)
    u_z, u_zz = ccd_z(cell_vell[0], dz)

    tensor_x = mu_z * u_z + mu_y * u_y + mu_x * u_x + mu * (u_xx + u_yy + u_zz)

    zero_order(vel[1], ghc, ndim)
    v_x, v_xx = ccd_x(cell_vell[1], dx)
    v_y, v_yy = ccd_y(cell_vell[1], dy)
    v_z, v_zz = ccd_z(cell_vell[1], dz)

    tensor_y = mu_z * v_z + mu_y * v_y + mu_x * v_x + mu * (v_xx + v_yy + v_zz)

    if ndim > 2:

        zero_order(vel[2], ghc, ndim)
        w_x, w_xx = ccd_x(cell_vell[2], dx)
        w_y, w_yy = ccd_y(cell_vell[2], dy)
        w_z, w_zz = ccd_z(cell_vell[2], dz)

        tensor_z = mu_z * w_z + mu_y * w_y + mu_x * w_x + mu * (w_xx + w_yy + w_zz)

    else:

        tensor_z = np.zeros_like(tensor_x)

    return np.stack((tensor_x, tensor_y, tensor_z))
