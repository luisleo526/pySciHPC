from pySciHPC.objects.sparse_matrix import SparseMatrix
from numba import cuda
import cupy as cp
from cupyx.scipy.sparse.linalg import spsolve


def sparse_matrix_bc(N: int, dx: float):
    m = SparseMatrix()
    m.add(1.0, 0, 0)  # A[1, 0] = 1.0
    m.add(2.0, 0, 1)  # A[2, 0] = 2.0
    m.add(-dx, 0, N + 1)  # B[2, 0] = -dx

    m.add(-2.5 / dx, N, 1)  # AA[2, 0] = -2.5 / dx
    m.add(1.0, N, N)  # BB[1, 0] = 1.0
    m.add(8.5, N, N + 1)  # BB[2, 0] = 8.5

    m.add(1.0, N - 1, N - 1)  # A[1, -1] = 1.0
    m.add(2.0, N - 1, N - 2)  # A[0, -1] = 2.0
    m.add(dx, N - 1, 2 * N - 2)  # B[0, -1] = dx

    m.add(2.5 / dx, 2 * N - 1, N - 2)  # AA[0, -1] = 2.5 / dx
    m.add(1.0, 2 * N - 1, 2 * N - 1)  # BB[1, -1] = 1.0
    m.add(8.5, 2 * N - 1, 2 * N - 2)  # BB[0, -1] = 8.5

    return m


def sparse_matrix(N: int, dx: float):
    """
    Construct (2N,2N) sparse matrix of coefficients
    """
    mu = sparse_matrix_bc(N, dx)
    md = sparse_matrix_bc(N, dx)

    a1: float = 0.875
    b1: float = 0.1251282341599089
    b2: float = -0.2487176584009104
    b3: float = 0.0001282341599089

    for i in range(1, N - 1):
        mu.add(a1, i, i - 1)
        mu.add(1.0, i, i)

        md.add(1.0, i, i)
        md.add(a1, i, i + 1)

        mu.add(b1 * dx, i, N + i - 1)
        mu.add(b2 * dx, i, N + i)
        mu.add(b3 * dx, i, N + i + 1)

        mu.add(-b3 * dx, i, N + i - 1)
        mu.add(-b2 * dx, i, N + i)
        mu.add(-b1 * dx, i, N + i + 1)

        mu.add(-9.0 / 8.0 / dx, N + i, i - 1)
        mu.add(9.0 / 8.0 / dx, N + i, i + 1)

        md.add(-9.0 / 8.0 / dx, N + i, i - 1)
        md.add(9.0 / 8.0 / dx, N + i, i + 1)

        mu.add(-1.0 / 8.0, N + i, N + i - 1)
        mu.add(1.0, N + i, N + i)
        mu.add(-1.0 / 8.0, N + i, N + i + 1)

        md.add(-1.0 / 8.0, N + i, N + i - 1)
        md.add(1.0, N + i, N + i)
        md.add(-1.0 / 8.0, N + i, N + i + 1)

    return mu.to_cupy(), md.to_cupy()


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

    SU[-1] = -(-3.5 * f[-1] + 4.0 * f[-2] - 0.5 * f[-3]) / dx
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


def cuda_UCCD(f: cp.ndarray, c: cp.ndarray, dx: float, blockdim: int, threaddim: int):
    N = f.shape[0]
    upwind, downwind = sparse_matrix(N, dx)
    su, sd = cuda_UCCD_src(f, dx, blockdim, threaddim)
    fx = cp.zeros_like(f)
    solu = spsolve(upwind, su)
    sold = spsolve(downwind, sd)
    fxu, fxxu = solu[:N], solu[N:]
    fxd, fxxd = sold[:N], sold[N:]

    retrieve_from_c[blockdim, threaddim](fx, fxu, fxd, c)

    del upwind, downwind, su, sd, solu, sold, fxu, fxxu, fxd, fxxd

    return fx
