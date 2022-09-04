import numpy as np
from numba import int32, float64, njit


def find_order(data: dict):
    Ns = list(data.keys())
    print(f"{'N':^15s}|{'L2 norm':^15s}|{'Order':^15s}")
    print("-" * 53)
    for i in range(len(Ns)):
        if i == 0:
            order = "-"
        else:
            order = abs((np.log(data[Ns[i]]) - np.log(data[Ns[i - 1]])) / np.log(Ns[i] / Ns[i - 1]))
            order = f"{order:^15.4f}"
        print(f"{Ns[i]:^15d}|{data[Ns[i]]:^15.6E}|{order:^15s}")


@njit(float64[:](float64[:], int32))
def pad(f: np.ndarray, N: int32):
    ff = np.zeros(f.size + 2 * N)
    for i in range(f.size):
        ff[i + N] = f[i]
    for i in range(N):
        ff[i] = f[0]
        ff[-i] = f[-1]
    return ff
