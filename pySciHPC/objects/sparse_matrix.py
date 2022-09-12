from cupyx.scipy.sparse import csc_matrix
from scipy.sparse import csc_array
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
        return csc_matrix((cp.array(self.data), (cp.array(self.row), cp.array(self.col))), dtype='float64')

    def to_numpy(self):
        return csc_array((self.data, (self.row, self.col)), dtype='float64')

    def clean(self):
        self.data = []
        self.col = []
        self.row = []
