import numpy as np

from lib.example_file import example_function


def test_example_function():
    """Check if the function gives the correct output"""
    inputs = [5, -7, 1.3, 0]
    outputs = [10, -14, 2.6, 0]
    computed_outputs = [example_function(x) for x in inputs]
    np.testing.assert_allclose(outputs, computed_outputs, err_msg="The example_function is not implemented correctly.")


if __name__ == '__main__':
    test_example_function()
    print("Test complete.")
