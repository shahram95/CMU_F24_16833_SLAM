import unittest
import numpy as np
from nonlinear import odometry_estimation, bearing_range_estimation, compute_meas_obs_jacobian, warp2pi

class TestSLAMFunctions(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0, 0, 1, 1, 3, 3, 2, 2])  # 3 poses and 1 landmark
        self.n_poses = 3

    def test_odometry_estimation(self):
        odom = odometry_estimation(self.x, 0)
        np.testing.assert_array_almost_equal(odom, np.array([1, 1]))

    def test_bearing_range_estimation(self):
        obs = bearing_range_estimation(self.x, 0, 0, self.n_poses)
        expected_theta = warp2pi(np.arctan2(2, 2))
        expected_distance = np.sqrt(8)
        np.testing.assert_array_almost_equal(obs, np.array([expected_theta, expected_distance]))

    def test_compute_meas_obs_jacobian(self):
        jacobian = compute_meas_obs_jacobian(self.x, 0, 0, self.n_poses)
        expected_jacobian = np.array([
            [0.25, -0.25, -0.25, 0.25],
            [-np.sqrt(2)/2, -np.sqrt(2)/2, np.sqrt(2)/2, np.sqrt(2)/2]
        ])
        np.testing.assert_array_almost_equal(jacobian, expected_jacobian)

if __name__ == '__main__':
    unittest.main()