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
