import numpy as np
from numba import njit, prange, float64, int32


@njit(float64[:, :, :, :](float64[:, :, :, :], int32), parallel=True, fastmath=True, nogil=True)
def all_face_x(stag_vector: np.ndarray, ndim: int):
    assert ndim > 1
    vector_face_x = np.copy(stag_vector)
    for i in prange(1, stag_vector[0].shape[0] - 1):
        for j in prange(1, stag_vector[0].shape[1] - 1):
            if ndim > 2:
                for k in prange(1, stag_vector[0].shape[2] - 1):
                    vector_face_x[1, i, j, k] = np.mean(stag_vector[1, i:i + 2, j - 1:j + 1, k])
                    vector_face_x[2, i, j, k] = np.mean(stag_vector[2, i:i + 2, j, k - 1:k + 1])
            else:
                k = 0
                vector_face_x[1, i, j, k] = np.mean(stag_vector[1, i:i + 2, j - 1:j + 1, k])

    return vector_face_x


@njit(float64[:, :, :, :](float64[:, :, :, :], int32), parallel=True, fastmath=True, nogil=True)
def all_face_y(stag_vector: np.ndarray, ndim: int):
    assert ndim > 1
    vector_face_y = np.copy(stag_vector)
    for i in prange(1, stag_vector[0].shape[0] - 1):
        for j in prange(1, stag_vector[0].shape[1] - 1):
            if ndim > 2:
                for k in prange(1, stag_vector[0].shape[2] - 1):
                    vector_face_y[0, i, j, k] = np.mean(stag_vector[0, i - 1:i + 1, j:j + 2, k])
                    vector_face_y[2, i, j, k] = np.mean(stag_vector[2, i, j:j + 2, k - 1:k + 1])
            else:
                k = 0
                vector_face_y[0, i, j, k] = np.mean(stag_vector[0, i - 1:i + 1, j:j + 2, k])

    return vector_face_y


@njit(float64[:, :, :, :](float64[:, :, :, :], int32), parallel=True, fastmath=True, nogil=True)
def all_face_z(stag_vector: np.ndarray, ndim: int):
    assert ndim > 2
    vector_face_z = np.copy(stag_vector)
    for i in prange(1, stag_vector[0].shape[0] - 1):
        for j in prange(1, stag_vector[0].shape[1] - 1):
            for k in prange(1, stag_vector[0].shape[2] - 1):
                vector_face_z[0, i, j, k] = np.mean(stag_vector[0, i - 1:i + 1, j, k:k + 2])
                vector_face_z[1, i, j, k] = np.mean(stag_vector[1, i, j - 1:j + 1, k:k + 2])
    return vector_face_z


@njit(float64[:, :, :, :](float64[:, :, :], int32), parallel=True, fastmath=True, nogil=True)
def cell_to_face(cell: np.ndarray, ndim: int):
    assert ndim > 1

    face_x = np.zeros_like(cell)
    face_y = np.zeros_like(cell)
    face_z = np.zeros_like(cell)

    for i in prange(cell.shape[0] - 1):
        for j in prange(cell.shape[1] - 1):
            if ndim > 2:
                for k in prange(max(cell.shape[2] - 1, 1)):
                    face_x[i, j, k] = (cell[i, j, k] + cell[i + 1, j, k]) / 2.0
                    face_y[i, j, k] = (cell[i, j, k] + cell[i, j + 1, k]) / 2.0
                    face_z[i, j, k] = (cell[i, j, k] + cell[i, j, k + 1]) / 2.0
                else:
                    k = 0
                    face_x[i, j, k] = (cell[i, j, k] + cell[i + 1, j, k]) / 2.0
                    face_y[i, j, k] = (cell[i, j, k] + cell[i, j + 1, k]) / 2.0

    return np.stack((face_x, face_y, face_z))


@njit(float64[:, :, :, :](float64[:, :, :, :], int32), parallel=True, fastmath=True, nogil=True)
def face_to_cell(stag_vector: np.ndarray, ndim: int):
    assert ndim > 1

    cell = np.zeros_like(stag_vector)

    for i in prange(1, cell[0].shape[0]):
        for j in prange(1, cell[0].shape[1]):
            if ndim > 2:
                for k in prange(1, cell[0].shape[2]):
                    cell[0, i, j, k] = (stag_vector[0, i, j, k] + stag_vector[0, i - 1, j, k]) / 2.0
                    cell[1, i, j, k] = (stag_vector[1, i, j, k] + stag_vector[1, i, j - 1, k]) / 2.0
                    cell[2, i, j, k] = (stag_vector[2, i, j, k] + stag_vector[2, i, j, k - 1]) / 2.0
            else:
                k = 0
                cell[0, i, j, k] = (stag_vector[0, i, j, k] + stag_vector[0, i - 1, j, k]) / 2.0
                cell[1, i, j, k] = (stag_vector[1, i, j, k] + stag_vector[1, i, j - 1, k]) / 2.0

    return cell
