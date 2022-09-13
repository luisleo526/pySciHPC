from pySciHPC.objects.sparse_matrix import SparseMatrix

"""
 A[j,i] ->   i, i+j-1
 B[j,i] ->   i, i+j-1+N
AA[j,i] -> N+i, i+j-1
BB[j,i] -> N+i, i+j-1+N
"""


def sparse_matrix_bc(N: int, dx: float):
    m = SparseMatrix()
    m.add(1.0, 0, 0)      # A[1, 0] = 1.0
    m.add(2.0, 0, 1)      # A[2, 0] = 2.0
    m.add(-dx, 0, N + 1)  # B[2, 0] = -dx

    m.add(-2.5 / dx, N, 1)  # AA[2, 0] = -2.5 / dx
    m.add(1.0, N, N)        # BB[1, 0] = 1.0
    m.add(8.5, N, N + 1)    # BB[2, 0] = 8.5

    m.add(1.0, N - 1, N - 1)     # A[1, -1] = 1.0
    m.add(2.0, N - 1, N - 2)     # A[0, -1] = 2.0
    m.add(dx, N - 1, 2 * N - 2)  # B[0, -1] = dx

    m.add(2.5 / dx, 2 * N - 1, N - 2)  # AA[0, -1] = 2.5 / dx
    m.add(1.0, 2 * N - 1, 2 * N - 1)   # BB[1, -1] = 1.0
    m.add(8.5, 2 * N - 1, 2 * N - 2)   # BB[0, -1] = 8.5

    return m


"""
 A[j,i] ->   i, i+j-1
 B[j,i] ->   i, i+j-1+N
AA[j,i] -> N+i, i+j-1
BB[j,i] -> N+i, i+j-1+N
"""


def CCD_sparse_matrix(N: int, dx: float):
    m = sparse_matrix_bc(N, dx)

    for i in range(1, N - 1):
        m.add(7.0 / 16.0, i, i - 1)
        m.add(1.0, i, i)
        m.add(7.0 / 16.0, i, i + 1)

        m.add(dx / 16.0, i, N + i - 1)
        m.add(-dx / 16.0, i, N + i + 1)

        m.add(-9.0 / 8.0 / dx, N + i, i - 1)
        m.add(9.0 / 8.0 / dx, N + i, i + 1)

        m.add(-1.0 / 8.0, N + i, N + i - 1)
        m.add(1.0, N + i, N + i)
        m.add(-1.0 / 8.0, N + i, N + i + 1)

    return m


"""
 A[j,i] ->   i, i+j-1
 B[j,i] ->   i, i+j-1+N
AA[j,i] -> N+i, i+j-1
BB[j,i] -> N+i, i+j-1+N
"""


def UCCD_sparse_matrix(N: int, dx: float):
    """
    Construct (2N,2N) sparse matrix of coefficients for UCCD scheme
    """
    mu = sparse_matrix_bc(N, dx)
    md = sparse_matrix_bc(N, dx)

    a1: float = 0.875
    b1: float = 0.1251282341599089
    b2: float = -0.2487176584009104
    b3: float = 0.0001282341599089

    for i in range(1, N - 1):
        mu.add(a1, i, i - 1)
        mu.add(1.0, i, i)

        md.add(1.0, i, i)
        md.add(a1, i, i + 1)

        mu.add(b1 * dx, i, N + i - 1)
        mu.add(b2 * dx, i, N + i)
        mu.add(b3 * dx, i, N + i + 1)

        md.add(-b3 * dx, i, N + i - 1)
        md.add(-b2 * dx, i, N + i)
        md.add(-b1 * dx, i, N + i + 1)

        mu.add(-9.0 / 8.0 / dx, N + i, i - 1)
        mu.add(9.0 / 8.0 / dx, N + i, i + 1)

        md.add(-9.0 / 8.0 / dx, N + i, i - 1)
        md.add(9.0 / 8.0 / dx, N + i, i + 1)

        mu.add(-1.0 / 8.0, N + i, N + i - 1)
        mu.add(1.0, N + i, N + i)
        mu.add(-1.0 / 8.0, N + i, N + i + 1)

        md.add(-1.0 / 8.0, N + i, N + i - 1)
        md.add(1.0, N + i, N + i)
        md.add(-1.0 / 8.0, N + i, N + i + 1)

    return mu, md
