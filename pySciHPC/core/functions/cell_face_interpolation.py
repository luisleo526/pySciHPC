import numpy as np
from numba import njit, prange, float64, int32


@njit(float64[:, :, :, :](float64[:, :, :, :], int32), parallel=True, fastmath=True, nogil=True)
def all_face_x(face: np.ndarray, ndim: int):
    assert ndim > 1
    int_face = np.copy(face)
    for i in prange(1, face[0].shape[0] - 1):
        for j in prange(1, face[0].shape[1] - 1):
            for k in prange(max(face[0].shape[2] - 1), 1):
                int_face[1, i, j, k] = np.mean(face[1, i:i + 2, j - 1:j + 1, k])
                if ndim > 2:
                    int_face[2, i, j, k] = np.mean(face[2, i:i + 2, j, k - 1:k + 1])
    return int_face


@njit(float64[:, :, :, :](float64[:, :, :, :], int32), parallel=True, fastmath=True, nogil=True)
def all_face_y(face: np.ndarray, ndim: int):
    assert ndim > 1
    int_face = np.copy(face)
    for i in prange(1, face[0].shape[0] - 1):
        for j in prange(1, face[0].shape[1] - 1):
            for k in prange(max(face[0].shape[2] - 1), 1):
                int_face[0, i, j, k] = np.mean(face[0, i - 1:i + 1, j:j + 2, k])
                if ndim > 2:
                    int_face[2, i, j, k] = np.mean(face[2, i, j:j + 2, k - 1:k + 1])
    return int_face


@njit(float64[:, :, :, :](float64[:, :, :, :], int32), parallel=True, fastmath=True, nogil=True)
def all_face_z(face: np.ndarray, ndim: int):
    assert ndim > 2
    int_face = np.copy(face)
    for i in prange(1, face[0].shape[0] - 1):
        for j in prange(1, face[0].shape[1] - 1):
            for k in prange(1, face[0].shape[2] - 1):
                int_face[0, i, j, k] = np.mean(face[0, i - 1:i + 1, j, k:k + 2])
                int_face[1, i, j, k] = np.mean(face[1, i, j - 1:j + 1, k:k + 2])
    return int_face
