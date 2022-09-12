from numba import cuda
import cupy as cp


@cuda.jit(device=True)
def cuda_WENO_weights_JS(b1, b2, b3):
    epsilon = 1.0e-14

    a1 = 1.0 / (epsilon + b1) ** 2
    a2 = 6.0 / (epsilon + b2) ** 2
    a3 = 3.0 / (epsilon + b3) ** 2

    w1 = a1 / (a1 + a2 + a3)
    w2 = a2 / (a1 + a2 + a3)
    w3 = a3 / (a1 + a2 + a3)

    return w1, w2, w3


@cuda.jit(device=True)
def cuda_WENO_weights_Z(b1, b2, b3):
    epsilon = 1.0e-14

    a1 = 1.0 * (1.0 + abs(b1 - b3) / (epsilon + b1))
    a2 = 6.0 * (1.0 + abs(b1 - b3) / (epsilon + b2))
    a3 = 3.0 * (1.0 + abs(b1 - b3) / (epsilon + b3))

    w1 = a1 / (a1 + a2 + a3)
    w2 = a2 / (a1 + a2 + a3)
    w3 = a3 / (a1 + a2 + a3)

    return w1, w2, w3


@cuda.jit(device=True)
def cuda_WENO_indicators_p(a, b, c, d, e):
    """
    a : i-1 | b: i | c: i+1 | d: i+2 | e: i+3
    """
    b3 = 13.0 * (a - 2.0 * b + c) ** 2 + 3.0 * (a - 4.0 * b + 3.0 * c) ** 2
    b2 = 13.0 * (b - 2.0 * c + d) ** 2 + 3.0 * (b - d) ** 2
    b1 = 13.0 * (c - 2.0 * d + e) ** 2 + 3.0 * (3.0 * c - 4.0 * d + e) ** 2

    return b1, b2, b3


@cuda.jit(device=True)
def cuda_WENO_indicators_m(a, b, c, d, e):
    """
    a : i-2 | b: i-1 | c: i | d: i+1 | e: i+2
    """
    b1 = 13.0 * (a - 2.0 * b + c) ** 2 + 3.0 * (a - 4.0 * b + 3.0 * c) ** 2
    b2 = 13.0 * (b - 2.0 * c + d) ** 2 + 3.0 * (b - d) ** 2
    b3 = 13.0 * (c - 2.0 * d + e) ** 2 + 3.0 * (3.0 * c - 4.0 * d + e) ** 2

    return b1, b2, b3


@cuda.jit
def cuda_WENO_JS_p(a, b, c, d, e, fp):
    """
    a : i-1 | b: i | c: i+1 | d: i+2 | e: i+3
    """
    i = cuda.grid(1)

    if i > 2 and i < a.shape[0] - 3:
        b1, b2, b3 = cuda_WENO_indicators_p(a[i], b[i], c[i], d[i], e[i])
        w1, w2, w3 = cuda_WENO_weights_JS(b1, b2, b3)

        f3 = (- a[i] + 5.0 * b[i] + 2.0 * c[i]) / 6.0
        f2 = (2.0 * b[i] + 5.0 * c[i] - d[i]) / 6.0
        f1 = (11.0 * c[i] - 7.0 * d[i] + 2.0 * e[i]) / 6.0

        fp[i] = w1 * f1 + w2 * f2 + w3 * f3


@cuda.jit
def cuda_WENO_JS_m(a, b, c, d, e, fm):
    """
    a : i-2 | b: i-1 | c: i | d: i+1 | e: i+2
    """

    i = cuda.grid(1)

    if i > 1 and i < a.shape[0] - 2:
        b1, b2, b3 = cuda_WENO_indicators_m(a[i], b[i], c[i], d[i], e[i])
        w1, w2, w3 = cuda_WENO_weights_JS(b1, b2, b3)

        f3 = (- e[i] + 5.0 * d[i] + 2.0 * c[i]) / 6.0
        f2 = (2.0 * d[i] + 5.0 * c[i] - b[i]) / 6.0
        f1 = (11.0 * c[i] - 7.0 * b[i] + 2.0 * a[i]) / 6.0

        fm[i] = w1 * f1 + w2 * f2 + w3 * f3


@cuda.jit
def cuda_WENO_Z_p(a, b, c, d, e, fp):
    """
    a : i-1 | b: i | c: i+1 | d: i+2 | e: i+3
    """
    i = cuda.grid(1)

    if i > 2 and i < a.shape[0] - 3:
        b1, b2, b3 = cuda_WENO_indicators_p(a[i], b[i], c[i], d[i], e[i])
        w1, w2, w3 = cuda_WENO_weights_Z(b1, b2, b3)

        f3 = (- a[i] + 5.0 * b[i] + 2.0 * c[i]) / 6.0
        f2 = (2.0 * b[i] + 5.0 * c[i] - d[i]) / 6.0
        f1 = (11.0 * c[i] - 7.0 * d[i] + 2.0 * e[i]) / 6.0

        fp[i] = w1 * f1 + w2 * f2 + w3 * f3


@cuda.jit
def cuda_WENO_Z_m(a, b, c, d, e, fm):
    """
    a : i-2 | b: i-1 | c: i | d: i+1 | e: i+2
    """

    i = cuda.grid(1)

    if i > 1 and i < a.shape[0] - 2:
        b1, b2, b3 = cuda_WENO_indicators_m(a[i], b[i], c[i], d[i], e[i])
        w1, w2, w3 = cuda_WENO_weights_Z(b1, b2, b3)

        f3 = (- e[i] + 5.0 * d[i] + 2.0 * c[i]) / 6.0
        f2 = (2.0 * d[i] + 5.0 * c[i] - b[i]) / 6.0
        f1 = (11.0 * c[i] - 7.0 * b[i] + 2.0 * a[i]) / 6.0

        fm[i] = w1 * f1 + w2 * f2 + w3 * f3


@cuda.jit
def cuda_flux(fp, fm, c, fh):
    i = cuda.grid(1)

    if i < fh.shape[0]:
        if c[i] > 0.0:
            fh[i] = fm[i]
        else:
            fh[i] = fp[i]


@cuda.jit
def cuda_derivatives_from_flux(fp, fm, fx, dx):
    i = cuda.grid(1)

    if i < fp.shape[0]:
        fx[i] = (fp[i] - fm[i]) / dx[i]


def cuda_WENO_JS(f: cp.ndarray, c: cp.ndarray, dx: float, blockdim: int, threaddim: int):
    fp = cp.zeros_like(f)
    fm = cp.zeros_like(f)
    fh = cp.zeros_like(f)
    fx = cp.zeros_like(f)
    dxs = cp.ones_like(f) * dx

    cuda_WENO_JS_p[blockdim, threaddim](cp.roll(f, 1), f, cp.roll(f, -1), cp.roll(f, -2), cp.roll(f, -3), fp)
    cuda_WENO_JS_m[blockdim, threaddim](cp.roll(f, 2), cp.roll(f, 1), f, cp.roll(f, -1), cp.roll(f, -2), fm)

    cuda_flux[blockdim, threaddim](fp, fm, c, fh)
    cuda_derivatives_from_flux[blockdim, threaddim](fh, cp.roll(fh, 1), fx, dxs)

    del fp, fm, fh, dxs

    return fx


def cuda_WENO_Z(f: cp.ndarray, c: cp.ndarray, dx: float, blockdim: int, threaddim: int, *args):
    fp = cp.zeros_like(f)
    fm = cp.zeros_like(f)
    fh = cp.zeros_like(f)
    fx = cp.zeros_like(f)
    dxs = cp.ones_like(f) * dx

    cuda_WENO_Z_p[blockdim, threaddim](cp.roll(f, 1), f, cp.roll(f, -1), cp.roll(f, -2), cp.roll(f, -3), fp)
    cuda_WENO_Z_m[blockdim, threaddim](cp.roll(f, 2), cp.roll(f, 1), f, cp.roll(f, -1), cp.roll(f, -2), fm)

    cuda_flux[blockdim, threaddim](fp, fm, c, fh)
    cuda_derivatives_from_flux[blockdim, threaddim](fh, cp.roll(fh, 1), fx, dxs)

    del fp, fm, fh, dxs

    return fx
