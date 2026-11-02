import numpy as np

from lib.eigendecomp import (
    get_inverse,
    get_dot_product,
    get_euclidean_norm,
    get_matrix_from_eigdec,
)


def main():
    print("---------- Define the input matrix.")
    A = 0.25 * np.array([[7, -np.sqrt(3)], [-np.sqrt(3), 5]])
    print(f"input matrix:\n{A}\n")

    print("---------- Do the eigendecomposition.")
    e, V = np.linalg.eig(A)
    print(f"Eigenvalues: {e}")
    print(f"Eigenvectors:\n{V}\n")

    # get the matrix back from the eigendecomposition
    A_ = get_matrix_from_eigdec(e, V)
    print(f"Restored matrix:\n{A_}\n")

    # check if matrices are close, they are not equal due to numerical inaccuracies
    np.testing.assert_allclose(A, A_, err_msg=f"Matrices are not equal: {A} and {A_}.")

    # get the 2 eigenvectors from the columns of the eigenvector matrix
    print("---------- Check orthonormality of the eigenvectors.")
    vec1, vec2 = V[:, 0], V[:, 1]

    # calculate norms of the vectors and check them
    norm1 = get_euclidean_norm(vec1)
    norm2 = get_euclidean_norm(vec2)
    np.testing.assert_allclose(
        [norm1, norm2],
        [1.0, 1.0],
        err_msg=f"Norms should be 1 but are {norm1} and {norm2}",
    )

    # calculate dot product of the vectors and check it
    dot = get_dot_product(vec1, vec2)
    np.testing.assert_allclose(
        [dot], [0.0], err_msg=f"Dot product should be 0 but is {dot}"
    )
    print("Columns are orthonormal.\n")

    # invert with numpy and your function and compare
    print("---------- Check get_inverse function.")
    ground_truth = np.linalg.inv(A)
    inverse = get_inverse(e, V)
    print(f"inverse with linalg:\n{ground_truth}\n")
    print(f"inverse with eigendecomposition:\n{inverse}\n")
    np.testing.assert_allclose(ground_truth, inverse, err_msg="Inverse is incorrect.")


if __name__ == "__main__":
    main()
