import argparse

import matplotlib.pyplot as plt
import numpy as np

from lib.norms import get_norm


def main():
    # create a script argument for the norm p, default p=2
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="norm (default 2)", type=float, default=2)
    args = parser.parse_args()
    p = args.p

    # compute and plot norms
    print(f"compute {p}-norm")
    X, Y, Z = get_norm(p)
    plot_norm(X, Y, Z, p)


def plot_norm(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, p: float) -> None:
    """
    Create 2D norm heatmap.

    Args:
        X: grid of x-axis values with shape (num_points, num_points)
        Y: grid of y-axis values with shape (num_points, num_points)
        Z: grid of norm values with shape (num_points, num_points)
        p: norm order (2 = euclidean)

    Returns:
        None
    """
    # setup figure
    plt.figure(figsize=(8, 8))

    # create the 3d contour plot
    plt.pcolor(X, Y, Z, cmap="plasma", shading="auto")

    # setup plot details
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Norm with p={p:.2f}")
    plt.grid()
    plt.axis("square")
    plt.colorbar()

    # show the plot
    plt.show()


if __name__ == "__main__":
    main()
