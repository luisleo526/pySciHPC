from cupyx.scipy.sparse import csr_matrix
from scipy.sparse import csr_array
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
        out = csr_matrix((cp.array(self.data), (cp.array(self.row), cp.array(self.col) )), dtype='float64')
        self.data = []
        self.col = []
        self.row = []
        return out

    def to_numpy(self):
        out = csr_array( (self.data, (self.row, self.col)), dtype='float64')
        self.data = []
        self.col = []
        self.row = []

        return out

