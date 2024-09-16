import numpy as np

from lib.eigendecomp import get_inverse


def test_inverse() -> np.ndarray:
    """Test inverse via eigendecomp"""
    input_matrix = 4.25 * np.array([[3, -np.sqrt(3)], [-np.sqrt(3), 5]])
    eigval, eigvec = np.linalg.eig(input_matrix)
    inverse_truth = np.linalg.inv(input_matrix)
    inverse_eig = get_inverse(eigval, eigvec)
    np.testing.assert_allclose(inverse_eig, inverse_truth, err_msg="")


if __name__ == '__main__':
    test_inverse()
    print("Test complete.")
