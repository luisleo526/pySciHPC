import numpy as np
from cupyx.scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from scipy.sparse import bmat, diags

from pySciHPC.scheme.spatial.UCCD import UCCD_coeffs

class UCCDMatrix:

    def __init__(self, shape: np.shape, grids: np.ndarray, use_cuda: bool = False):
        self.matrix = []
        for i in range(grids.size):
            if grids[i] > 0:
                AU, BU, AAU, BBU, AD, BD, AAD, BBD = UCCD_coeffs(shape[i], grids[i])

                AU = diags([AU[0, 1:], AU[1, :], AU[2, :-1]], [-1, 0, 1], format='csc')
                BU = diags([BU[0, 1:], BU[1, :], BU[2, :-1]], [-1, 0, 1], format='csc')
                AAU = diags([AAU[0, 1:], AAU[1, :], AAU[2, :-1]], [-1, 0, 1], format='csc')
                BBU = diags([BBU[0, 1:], BBU[1, :], BBU[2, :-1]], [-1, 0, 1], format='csc')

                AD = diags([AD[0, 1:], AD[1, :], AD[2, :-1]], [-1, 0, 1], format='csc')
                BD = diags([BD[0, 1:], BD[1, :], BD[2, :-1]], [-1, 0, 1], format='csc')
                AAD = diags([AAD[0, 1:], AAD[1, :], AAD[2, :-1]], [-1, 0, 1], format='csc')
                BBD = diags([BBD[0, 1:], BBD[1, :], BBD[2, :-1]], [-1, 0, 1], format='csc')

                imu = csc_matrix(bmat([[AU, BU], [AAU, BBU]], format='csc'))
                imd = csc_matrix(bmat([[AD, BD], [AAD, BBD]], format='csc'))

            else:
                imu = imd = None

            self.matrix.append((imu, imd))
