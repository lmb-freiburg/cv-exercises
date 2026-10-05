"""Norm computing."""

from typing import Tuple

import numpy as np


def get_norm(
    p: float = 2, num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute norms of 2D vectors resulting from combining the i-th entry of two matrices X and Y.

    Args:
        p: norm order (2 = euclidean)
        num_points: pixel resolution of the data

    Returns:
        Tuple of:
            X: coordinates for each point on the grid (X(i,j)) with shape (num_points, num_points)
            Y: coordinates for each point on the grid (X(i,j)) with shape (num_points, num_points)
            Z: Norm for each point in the grid with shape (num_points, num_points)

    Note:
        The shape of the Z array this function returns should be the same as the shapes of X and Y.
    """
    # Create 2d input values with meshgrid
    lin_x = np.linspace(-10, 10, num_points)
    lin_y = np.linspace(-10, 10, num_points)
    X, Y = np.meshgrid(lin_x, lin_y)

    # X: grid of x-axis values with shape (num_points, num_points)
    # Y: grid of y-axis values with shape (num_points, num_points)

    # START TODO #################
    # stack the two inputs at the last axis, then compute the norm over that last axis.
    XY = np.stack((X, Y), axis=-1)
    Z = np.linalg.norm(XY, ord=p, axis=-1)
    # END TODO ###################

    return X, Y, Z
