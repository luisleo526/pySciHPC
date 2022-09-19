from ...core.data import Scalar


def cuda_periodic(f: Scalar):
    for j in range(f.data.gpu[0].shape[1]):
        for k in range(f.data.gpu[0].shape[2]):
            for i in range(f.ghc):
                f.data.gpu[0, i, j, k] = f.data.gpu[0, -2 * f.ghc + i, j, k]
                f.data.gpu[0, -i - 1, j, k] = f.data.gpu[0, 2 * f.ghc - i - 1, j, k]

    if f.ndim > 1:
        for i in range(f.data.gpu[0].shape[0]):
            for k in range(f.data.gpu[0].shape[2]):
                for j in range(f.ghc):
                    f.data.gpu[0, i, j, k] = f.data.gpu[0, i, -2 * f.ghc + j, k]
                    f.data.gpu[0, i, -j - 1, k] = f.data.gpu[0, i, 2 * f.ghc - j - 1, k]

    if f.ndim > 2:
        for j in range(f.data.gpu[0].shape[1]):
            for i in range(f.data.gpu[0].shape[0]):
                for k in range(f.ghc):
                    f.data.gpu[0, i, j, k] = f.data.gpu[0, i, j, -2 * f.ghc + k]
                    f.data.gpu[0, i, j, -k - 1] = f.data.gpu[0, i, j, 2 * f.ghc - k - 1]
