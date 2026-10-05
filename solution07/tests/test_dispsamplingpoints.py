import numpy as np

from lib.corr import DispSamplingPoints


def test_dispsamplingpoints():
    """Test dot product of the columns of a 2x2 matrix"""
    points1_pred = DispSamplingPoints().offsets.detach().cpu().numpy()
    points1_gt = np.array(
        [
            [-40, 0],
            [-39, 0],
            [-38, 0],
            [-37, 0],
            [-36, 0],
            [-35, 0],
            [-34, 0],
            [-33, 0],
            [-32, 0],
            [-31, 0],
            [-30, 0],
            [-29, 0],
            [-28, 0],
            [-27, 0],
            [-26, 0],
            [-25, 0],
            [-24, 0],
            [-23, 0],
            [-22, 0],
            [-21, 0],
            [-20, 0],
            [-19, 0],
            [-18, 0],
            [-17, 0],
            [-16, 0],
            [-15, 0],
            [-14, 0],
            [-13, 0],
            [-12, 0],
            [-11, 0],
            [-10, 0],
            [-9, 0],
            [-8, 0],
            [-7, 0],
            [-6, 0],
            [-5, 0],
            [-4, 0],
            [-3, 0],
            [-2, 0],
            [-1, 0],
            [0, 0],
        ]
    )
    np.testing.assert_allclose(points1_gt, points1_pred)

    points2_pred = (
        DispSamplingPoints(steps=10, step_size=0.3).offsets.detach().cpu().numpy()
    )
    points2_gt = np.array(
        [
            [-3.0, 0.0],
            [-2.7, 0.0],
            [-2.4, 0.0],
            [-2.1, 0.0],
            [-1.8, 0.0],
            [-1.5, 0.0],
            [-1.2, 0.0],
            [-0.9, 0.0],
            [-0.6, 0.0],
            [-0.3, 0.0],
            [0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(points2_gt, points2_pred)


if __name__ == "__main__":
    test_dispsamplingpoints()
    print("Test complete.")
