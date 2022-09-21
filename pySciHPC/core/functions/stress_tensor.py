import numpy as np
from numba import njit, float64, int32

from .derivatives import find_fx_fxx, find_fy_fyy, find_fz_fzz
from ..scheme.spatial import CCD_full
from ..functions.cell_face_interpolation import face_to_cell
from ..boundary_conditions.cell import zero_order_all, zero_order


@njit(float64[:, :, :, :](float64[:, :, :, :], float64[:], int32, int32, float64[:, :, :]), fastmath=True,
      parallel=True, nogil=True)
def ccd_stress(vel: np.ndarray, grids: np.ndarray, ghc: int, ndim: int, mu: np.ndarray):
    cell_vel = face_to_cell(vel, ndim)
    zero_order_all(cell_vel)
    zero_order(mu)

    dx, dy, dz = grids

    mu_x, mu_xx = find_fx_fxx(mu, dx, CCD_full)
    mu_y, mu_yy = find_fy_fyy(mu, dy, CCD_full)
    mu_z, mu_zz = find_fz_fzz(mu, dz, CCD_full)

    u_x, u_xx = find_fx_fxx(cell_vel[0], dx, CCD_full)
    u_y, u_yy = find_fy_fyy(cell_vel[0], dy, CCD_full)
    u_z, u_zz = find_fz_fzz(cell_vel[0], dz, CCD_full)

    tensor_x = mu_z * u_z + mu_y * u_y + mu_x * u_x + mu * (u_xx + u_yy + u_zz)

    v_x, v_xx = find_fx_fxx(cell_vel[1], dx, CCD_full)
    v_y, v_yy = find_fy_fyy(cell_vel[1], dy, CCD_full)
    v_z, v_zz = find_fz_fzz(cell_vel[1], dz, CCD_full)

    tensor_y = mu_z * v_z + mu_y * v_y + mu_x * v_x + mu * (v_xx + v_yy + v_zz)

    if ndim > 2:

        w_x, w_xx = find_fx_fxx(cell_vel[2], dx, CCD_full)
        w_y, w_yy = find_fy_fyy(cell_vel[2], dy, CCD_full)
        w_z, w_zz = find_fz_fzz(cell_vel[2], dz, CCD_full)

        tensor_z = mu_z * w_z + mu_y * w_y + mu_x * w_x + mu * (w_xx + w_yy + w_zz)

    else:

        tensor_z = np.zeros_like(tensor_x)

    return np.stack((tensor_x, tensor_y, tensor_z))
