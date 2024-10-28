"""Eigendecomposition functions."""

import numpy as np


def get_matrix_from_eigdec(e: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Restore the original square symmetric matrix from eigenvalues and eigenvectors after eigenvalue decomposition.

    Args:
        e: The vector of eigenvalues with shape (N).
        V: The matrix with eigenvectors as columns with shape (N, N).

    Returns:
        The original matrix used for eigenvalue decomposition with shape (N, N)
    """
    # START TODO #################
    return np.matmul(V, np.matmul(np.diag(e), V.T))  # A = QwQ^{T}
    # END TODO ###################


def get_euclidean_norm(v: np.ndarray) -> np.ndarray:
    """Compute the euclidean norm of a vector.

    Args:
        v (np.ndarray): The input vector with shape (vector_length).

    Returns:
        The euclidean norm of the vector.
    """
    # START TODO #################
    # Do not use np.linalg.norm
    norm = np.sqrt(np.sum(v**2))
    return norm
    # END TODO ###################


def get_dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute dot product of two vectors.

    Args:
        v1: First input vector with shape (vector_length)
        v2: Second input vector with shape (vector_length)

    Returns:
        Dot product result.
    """
    assert (
        len(v1.shape) == len(v2.shape) == 1 and v1.shape == v2.shape
    ), f"Input vectors must be 1-dimensional and have the same shape, but have shapes {v1.shape} and {v2.shape}"
    # START TODO #################
    return np.dot(v1, v2)
    # END TODO ###################


def get_inverse(e: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute the inverse of a square symmetric matrix A given its eigenvectors and eigenvalues.

    Args:
        e: The vector of eigenvalues with shape (N).
        V: The matrix with eigenvectors as columns with shape (N, N).

    Returns:
        The inverse of A (i.e. the matrix with given eigenvalues/vectors) with shape (N, N).
    """
    # START TODO #################
    # Do not use np.linalg.inv
    return np.matmul(V, np.matmul(np.diag(1 / e), V.T))  # A^{-1} = Qw^{-1}Q^{T}
    # END TODO ###################
