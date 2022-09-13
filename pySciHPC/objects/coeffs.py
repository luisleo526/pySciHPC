import numpy as np
from cupyx.scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv

from pySciHPC.scheme.spatial.CCD_sparse_matrix import CCD_sparse_matrix, UCCD_sparse_matrix


class CCDMatrix:

    def __init__(self, shape: np.shape, grids: np.ndarray):
        self.matrix = []
        for i in range(grids.size):
            if grids[i] > 0:
                m = CCD_sparse_matrix(shape[i], grids[i])
                im = csc_matrix(inv(m.to_numpy()))
            else:
                im = None
            self.matrix.append(im)


class UCCDMatrix:

    def __init__(self, shape: np.shape, grids: np.ndarray, use_cuda: bool = False):
        self.matrix = []
        for i in range(grids.size):
            if grids[i] > 0:
                mu, md = UCCD_sparse_matrix(shape[i], grids[i])
                # imu = csc_matrix(inv(mu.to_numpy())).toarray()
                # imd = csc_matrix(inv(md.to_numpy())).toarray()
                imu = mu.to_cupy()
                imd = md.to_cupy()
            else:
                imu = imd = None

            self.matrix.append((imu, imd))
