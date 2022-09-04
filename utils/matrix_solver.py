import numpy as np

from numba import njit, float64


@njit(float64[:](float64[:], float64[:], float64[:], float64[:]))
def TDMA(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    N = a.size

    cp = np.zeros(N, dtype='float64')
    dp = np.zeros(N, dtype='float64')
    X = np.zeros(N, dtype='float64')

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in np.arange(1, N, 1):
        dnum = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / dnum
        dp[i] = (d[i] - a[i] * dp[i - 1]) / dnum

    X[N - 1] = dp[N - 1]

    for i in np.arange(N - 2, -1, -1):
        X[i] = dp[i] - cp[i] * X[i + 1]

    return X


@njit((float64[:, :], float64[:, :], float64[:, :], float64[:, :]))
def twin_dec(a: np.ndarray, b: np.ndarray, aa: np.ndarray, bb: np.ndarray):
    '''
    Decompose the coefficient matrix  [[A, B], [AA, BB]]
    '''

    N = a.shape[1]

    a[0, 0] = 0.0
    aa[0, 0] = 0.0
    a[2, N-1] = 0.0
    aa[2, N-1] = 0.0

    b[0, 0] = 0.0
    bb[0, 0] = 0.0
    b[2, N-1] = 0.0
    bb[2, N-1] = 0.0

    for i in range(1, N):
        den = a[1, i - 1] * bb[1, i - 1] - aa[1, i - 1] * b[1, i - 1]
        sc1 = -(aa[1, i - 1] * b[0, i] - a[0, i] * bb[1, i - 1]) / den
        sc2 = (b[0, i] * a[1, i - 1] - b[1, i - 1] * a[0, i]) / den

        a[0, i] = sc1
        a[1, i] = a[1, i] - (sc1 * a[2, i - 1] + sc2 * aa[2, i - 1])

        b[0, i] = sc2
        b[1, i] = b[1, i] - (sc1 * b[2, i - 1] + sc2 * bb[2, i - 1])

        sc1 = -(aa[1, i - 1] * bb[0, i] - aa[0, i] * bb[1, i - 1]) / den
        sc2 = (bb[0, i] * a[1, i - 1] - b[1, i - 1] * aa[0, i]) / den

        aa[0, i] = sc1
        aa[1, i] = aa[1, i] - (sc1 * a[2, i - 1] + sc2 * aa[2, i - 1])

        bb[0, i] = sc2
        bb[1, i] = bb[1, i] - (sc1 * b[2, i - 1] + sc2 * bb[2, i - 1])


@njit(float64[:, :](float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:]))
def twin_bks(a: np.ndarray, b: np.ndarray, aa: np.ndarray, bb: np.ndarray, s: np.ndarray, ss: np.ndarray):
    '''
    Solve for X, Y using decomposed coefficient matrix -> [[A, B],[AA,BB]] * [[X], [Y]] = [[S],[SS]]
    '''

    N = a.shape[1]
    for i in range(1, N):
        s[i] = s[i] - (a[0, i] * s[i - 1] + b[0, i] * ss[i - 1])
        ss[i] = ss[i] - (aa[0, i] * s[i - 1] + bb[0, i] * ss[i - 1])

    den = a[1, N-1] * bb[1, N-1] - aa[1, N-1] * b[1, N-1]
    sols = - (b[1, N-1] * ss[N-1] - bb[1, N-1] * s[N-1]) / den
    solss = (a[1, N-1] * ss[N-1] - aa[1, N-1] * s[N-1]) / den

    s[N-1] = sols
    ss[N-1] = solss

    for i in range(N - 2, -1, -1):
        s[i] = s[i] - (a[2, i] * s[i+1] + b[2, i] * ss[i+1])
        ss[i] = ss[i] - (aa[2, i] * s[i+1] + bb[2, i] * ss[i+1])

        den = a[1, i] * bb[1, i] - aa[1, i] * b[1, i]
        sols = - (b[1, i] * ss[i] - bb[1, i] * s[i]) / den
        solss = (a[1, i] * ss[i] - aa[1, i] * s[i]) / den

        s[i] = sols
        ss[i] = solss

    return np.stack((s, ss))
