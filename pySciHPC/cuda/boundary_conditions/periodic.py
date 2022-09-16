from ...core.data import Scalar


def cuda_periodic(f: Scalar, geo: Scalar):

    for j in range(f.data.gpu[0].shape[1]):
        for k in range(f.data.gpu[0].shape[2]):
            for i in range(geo.ghc):
                f.data.gpu[0, i, j, k] = f.data.gpu[0, -2 * geo.ghc - 1 + i, j, k]
                f.data.gpu[0, -i - 1, j, k] = f.data.gpu[0, 2 * geo.ghc - i, j, k]

    if geo.ndim > 1:
        for i in range(f.data.gpu[0].shape[0]):
            for k in range(f.data.gpu[0].shape[2]):
                for j in range(geo.ghc):
                    f.data.gpu[0, i, j, k] = f.data.gpu[0, i, -2 * geo.ghc - 1 + j, k]
                    f.data.gpu[0, i, -j - 1, k] = f.data.gpu[0, i, 2 * geo.ghc - j, k]

    if geo.ndim > 2:
        for j in range(f.data.gpu[0].shape[1]):
            for i in range(f.data.gpu[0].shape[0]):
                for k in range(geo.ghc):
                    f.data.gpu[0, i, j, k] = f.data.gpu[0, i, j, -2 * geo.ghc - 1 + k]
                    f.data.gpu[0, i, j, -k - 1] = f.data.gpu[0, i, j, 2 * geo.ghc - k]
