from cupyx.scipy.sparse import csc_matrix as cp_sparse
from scipy.sparse import csc_matrix as np_sparse
import cupy as cp


class SparseMatrix:

    def __init__(self):
        self.data = []
        self.col = []
        self.row = []

    def add(self, value, i, j):
        self.data.append(value)
        self.row.append(i)
        self.col.append(j)

    def to_cupy(self):
        return cp_sparse((cp.array(self.data), (cp.array(self.row), cp.array(self.col))), dtype='float64')

    def to_numpy(self):
        return np_sparse((self.data, (self.row, self.col)), dtype='float64')

    def clean(self):
        self.data = []
        self.col = []
        self.row = []
