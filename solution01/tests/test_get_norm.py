import numpy as np

from lib.norms import get_norm


def test_get_norm():
    """Test norms"""
    inputs = get_norm(p=3, num_points=5)
    truths = [np.array([[-10., -5., 0., 5., 10.],
                        [-10., -5., 0., 5., 10.],
                        [-10., -5., 0., 5., 10.],
                        [-10., -5., 0., 5., 10.],
                        [-10., -5., 0., 5., 10.]]),
              np.array([[-10., -10., -10., -10., -10.],
                        [-5., -5., -5., -5., -5.],
                        [0., 0., 0., 0., 0.],
                        [5., 5., 5., 5., 5.],
                        [10., 10., 10., 10., 10.]]),
              np.array([[12.5992105, 10.40041912, 10., 10.40041912, 12.5992105],
                        [10.40041912, 6.29960525, 5., 6.29960525, 10.40041912],
                        [10., 5., 0., 5., 10.],
                        [10.40041912, 6.29960525, 5., 6.29960525, 10.40041912],
                        [12.5992105, 10.40041912, 10., 10.40041912, 12.5992105]])]
    for inp, tru in zip(inputs, truths):
        np.testing.assert_allclose(inp, tru, err_msg="Norm computation is not implemented correctly.")


if __name__ == '__main__':
    test_get_norm()
    print("Test complete.")
