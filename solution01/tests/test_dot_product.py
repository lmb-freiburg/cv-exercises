import numpy as np

from lib.eigendecomp import get_dot_product


def test_dot_product() -> np.ndarray:
    """Test dot product of the columns of a 2x2 matrix"""
    input_matrix = 4.25 * np.array([[3, -np.sqrt(3)], [-np.sqrt(3), 5]])
    v1, v2 = input_matrix[:, 0], input_matrix[:, 1]
    dot = get_dot_product(v1, v2)
    assert isinstance(dot, float), f"Dot product should be a float but is {type(dot)}"
    np.testing.assert_allclose(dot, -250.28134169370276)


if __name__ == '__main__':
    test_dot_product()
    print("Test complete.")
