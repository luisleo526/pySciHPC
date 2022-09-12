from pySciHPC.scheme.spatial.CCD_sparse_matrix import CCD_sparse_matrix, UCCD_sparse_matrix
from cupyx.scipy.sparse import csc_matrix
import numpy as np
from scipy.sparse.linalg import inv
from numba.typed import List


class CCDMatrix:

    def __init__(self, shape: np.shape, grids: np.ndarray, use_cuda: bool = False):
        self.matrix = List()
        for i in range(grids.size):
            if grids[i] > 0:
                m = CCD_sparse_matrix(shape[i], grids[i])
                im = inv(m.to_numpy())
                if use_cuda:
                    im = csc_matrix(im)
                else:
                    im = im.toarray()
            else:
                im = np.zeros((2, 2), dtype='float64')
            self.matrix.append(im)


class UCCDMatrix:

    def __init__(self, shape: np.shape, grids: np.ndarray, use_cuda: bool = False):
        self.matrix = List()
        for i in range(grids.size):
            if grids[i] > 0:
                mu, md = UCCD_sparse_matrix(shape[i], grids[i])
                imu = inv(mu.to_numpy())
                imd = inv(md.to_numpy())
                if use_cuda:
                    imu = csc_matrix(imu)
                    imd = csc_matrix(imd)
                else:
                    imu = imu.toarray()
                    imd = imd.toarray()
            else:
                imu = np.zeros((2, 2), dtype='float64')
                imd = np.zeros((2, 2), dtype='float64')

            self.matrix.append(np.stack((imu, imd)))
