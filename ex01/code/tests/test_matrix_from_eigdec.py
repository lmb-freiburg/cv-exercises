import numpy as np

from lib.eigendecomp import get_matrix_from_eigdec


def test_matrix_from_eigdec() -> np.ndarray:
    """Test matrix reconstruction from eigendecomposition"""
    input_matrix = 7 * np.array([[7, -np.sqrt(3)], [-np.sqrt(3), 5]])
    eigval, eigvec = np.linalg.eig(input_matrix)
    restored_matrix = get_matrix_from_eigdec(eigval, eigvec)
    np.testing.assert_allclose(
        input_matrix, restored_matrix,
        err_msg="Matrix reconstruction from eigendecomposition is not correctly implemented.")


if __name__ == '__main__':
    test_matrix_from_eigdec()
    print("Test complete.")
