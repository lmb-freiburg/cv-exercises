import numpy as np

from lib.eigendecomp import get_euclidean_norm


def test_euclidean_norm():
    """Test euclidean norm."""
    v1 = np.array([-7, 5])
    norm = get_euclidean_norm(v1)
    assert isinstance(norm, float), f"Norm should be a float but is {type(norm)}"
    np.testing.assert_allclose(norm, 8.60232527, err_msg="Norm is not correctly implemented.")


if __name__ == '__main__':
    test_euclidean_norm()
    print("Test complete.")
