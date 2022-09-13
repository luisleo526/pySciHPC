import cupy as cp
from numba import cuda
from cupyx.scipy.sparse.linalg import spsolve

@cuda.jit
def cuda_UCCD_src_kernel(f, su, ssu, sd, ssd, dx):
    i = cuda.grid(1)

    c3: float = -0.06096119008109
    c2: float = 1.99692238016218
    c1: float = -1.93596119008109

    if i > 0 and i < f.shape[0] - 1:
        su[i] = (c1 * f[i - 1] + c2 * f[i] + c3 * f[i + 1]) / dx[i]
        ssu[i] = (3.0 * f[i - 1] - 6.0 * f[i] + 3.0 * f[i + 1]) / dx[i] ** 2

        sd[i] = -(c3 * f[i - 1] + c2 * f[i] + c1 * f[i + 1]) / dx[i]
        ssd[i] = (3.0 * f[i - 1] - 6.0 * f[i] + 3.0 * f[i + 1]) / dx[i] ** 2


def cuda_UCCD_src(f: cp.ndarray, dx: float, blockdim: int, threaddim: int):

    SU = cp.zeros_like(f)
    SD = cp.zeros_like(f)
    SSU = cp.zeros_like(f)
    SSD = cp.zeros_like(f)
    dxs = cp.ones_like(f) * dx

    cuda_UCCD_src_kernel[blockdim, threaddim](f, SU, SSU, SD, SSD, dxs)

    SU[0] = (-3.5 * f[0] + 4.0 * f[1] - 0.5 * f[2]) / dx
    SSU[0] = (34.0 / 3.0 * f[0] - 83.0 / 4.0 * f[1] + 10.0 * f[2] - 7.0 / 12.0 * f[3]) / dx ** 2

    SU[-1] = (-3.5 * f[-1] + 4.0 * f[-2] - 0.5 * f[-3]) / dx
    SSU[-1] = (34.0 / 3.0 * f[-1] - 83.0 / 4.0 * f[-2] + 10.0 * f[-3] - 7.0 / 12.0 * f[-4]) / dx ** 2

    SD[0] = (-3.5 * f[0] + 4.0 * f[1] - 0.5 * f[2]) / dx
    SSD[0] = (34.0 / 3.0 * f[0] - 83.0 / 4.0 * f[1] + 10.0 * f[2] - 7.0 / 12.0 * f[3]) / dx ** 2

    SD[-1] = -(-3.5 * f[-1] + 4.0 * f[-2] - 0.5 * f[-3]) / dx
    SSD[-1] = (34.0 / 3.0 * f[-1] - 83.0 / 4.0 * f[-2] + 10.0 * f[-3] - 7.0 / 12.0 * f[-4]) / dx ** 2

    del dxs

    return cp.concatenate((SU, SSU)), cp.concatenate((SD, SSD))


@cuda.jit
def retrieve_from_c(fx, fxu, fxd, c):
    i = cuda.grid(1)
    if i < fx.shape[0]:
        if c[i] > 0.0:
            fx[i] = fxd[i]
        else:
            fx[i] = fxu[i]


def cuda_UCCD(f: cp.ndarray, c: cp.ndarray, dx: float, blockdim: int, threaddim: int, coeff):
    N = f.shape[0]

    imu, imd = coeff
    su, sd = cuda_UCCD_src(f, dx, blockdim, threaddim)

    fx = cp.zeros_like(f)

    # solu = imu.dot(su)
    # sold = imd.dot(sd)

    solu = spsolve(imu, su)
    sold = spsolve(imd, sd)

    fxu, fxxu = solu[:N], solu[N:]
    fxd, fxxd = sold[:N], sold[N:]

    retrieve_from_c[blockdim, threaddim](fx, fxu, fxd, c)

    del su, sd, solu, sold, fxu, fxxu, fxd, fxxd

    return fx
