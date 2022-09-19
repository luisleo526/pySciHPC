import numpy as np
from numba import njit, float64, int32

from .derivatives import ccd_x, ccd_y, ccd_z


@njit(float64[:, :, :, :](float64[:, :, :, :], float64[:], int32, float64[:, :, :]), fastmath=True, parallel=True,
      nogil=True)
def CCD_stress(vel: np.ndarray, grids: np.ndarray, ndim: int, mu: np.ndarray):
    dx, dy, dz = grids

    mu_x, mu_xx = ccd_x(mu, dx)
    mu_y, mu_yy = ccd_y(mu, dy)
    mu_z, mu_zz = ccd_z(mu, dz)

    u_x, u_xx = ccd_x(vel[0], dx)
    u_y, u_yy = ccd_y(vel[0], dy)
    u_z, u_zz = ccd_z(vel[0], dz)

    tensor_x = mu_z * u_z + mu_y * u_y + mu_x * u_x + mu * (u_xx + u_yy + u_zz)

    v_x, v_xx = ccd_x(vel[1], dx)
    v_y, v_yy = ccd_y(vel[1], dy)
    v_z, v_zz = ccd_z(vel[1], dz)

    tensor_y = mu_z * v_z + mu_y * v_y + mu_x * v_x + mu * (v_xx + v_yy + v_zz)

    if ndim > 2:

        w_x, w_xx = ccd_x(vel[2], dx)
        w_y, w_yy = ccd_y(vel[2], dy)
        w_z, w_zz = ccd_z(vel[2], dz)

        tensor_z = mu_z * w_z + mu_y * w_y + mu_x * w_x + mu * (w_xx + w_yy + w_zz)

    else:

        tensor_z = np.zeros_like(tensor_x)

    return np.stack((tensor_x, tensor_y, tensor_z))
