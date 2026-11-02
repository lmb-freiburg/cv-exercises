import numpy as np


def main():
    # create input matrices
    A = np.array(np.arange(4))
    B = np.array([-1, 3])

    print("\n---------- Input matrices shape, type and content:")
    print(f"A (shape: {A.shape}, type: {type(A)}) = {A}")
    print(f"B (shape: {B.shape}, type: {type(B)}) = {B}")

    print("\n---------- Matrix multiplication on those matrices does not work:")
    try:
        np.matmul(A, B)
    except ValueError as e:
        print(f"Error type: {type(e)} message: {e}")

    print("\n---------- Reshape A from shape (4,) to shape (2,2) and store it in C:")
    C = A.reshape([2, 2])
    print(f"C shape: {C.shape}, content:\n{C}")

    print("\n---------- Now matrix multiplication of C (2,2) with B (2,) works:")
    matmul_result = np.matmul(C, B)
    print(f"Matrix multiplication result:\n{matmul_result}")

    print(
        "\n---------- When adding C (2,2) and B (2,), B will be automatically "
        "broadcasted to match the shape of A."
    )
    add_result = np.add(C, B)
    print(f"Addition result:\n{add_result}")

    print(
        "\n---------- The star operator (*) will do an element-wise multiplication between C (2,2) and B (2,). "
        "Again, B will be broadcasted to fit the shape of C. "
    )
    print(f"Element-wise multiplication result:\n{C * B}")

    print(
        "\n---------- np.diag can transform the vector B (2,) into a diagonal matrix of shape (2,2)"
    )
    print(f"B as diagonal matrix:\n{np.diag(B)}")

    print("\n---------- Use np.transpose to transpose the C")
    print(f"Transpose result:\n{np.transpose(C)}")


if __name__ == "__main__":
    main()
