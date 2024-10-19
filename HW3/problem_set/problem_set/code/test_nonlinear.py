import unittest
import numpy as np
from scipy.sparse import csr_matrix
from nonlinear import odometry_estimation, bearing_range_estimation, compute_meas_obs_jacobian, warp2pi, create_linear_system

class TestSLAMFunctions(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0, 0, 1, 1, 3, 3, 2, 2])  # 3 poses and 1 landmark
        self.n_poses = 3
        self.n_landmarks = 1
        self.odoms = np.array([[1, 1], [2, 2]])
        self.observations = np.array([[0, 0, np.pi/4, np.sqrt(8)], 
                                      [1, 0, np.arctan2(1, 1), np.sqrt(2)]])
        self.sigma_odom = np.eye(2)
        self.sigma_observation = np.eye(2)

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

    def test_create_linear_system(self):
        A, b = create_linear_system(self.x, self.odoms, self.observations, 
                                    self.sigma_odom, self.sigma_observation, 
                                    self.n_poses, self.n_landmarks)

        # Check dimensions
        expected_M = (len(self.odoms) + 1) * 2 + len(self.observations) * 2
        expected_N = self.n_poses * 2 + self.n_landmarks * 2
        self.assertEqual(A.shape, (expected_M, expected_N))
        self.assertEqual(b.shape, (expected_M,))

        # Check if A is sparse
        self.assertIsInstance(A, csr_matrix)

        # Check prior on first pose
        self.assertTrue(np.allclose(A.toarray()[:2, :2], np.eye(2)))

        # Check if b is not all zeros (should contain residuals)
        self.assertFalse(np.allclose(b, np.zeros_like(b)))

        # Check odometry residuals
        for i in range(len(self.odoms)):
            odom_error = self.odoms[i] - odometry_estimation(self.x, i)
            self.assertTrue(np.allclose(b[2*(i+1):2*(i+2)], odom_error, atol=1e-6))

        # Check observation residuals
        for i, obs in enumerate(self.observations):
            pose_idx, landmark_idx = int(obs[0]), int(obs[1])
            predicted = bearing_range_estimation(self.x, pose_idx, landmark_idx, self.n_poses)
            error = obs[2:] - predicted
            error[0] = warp2pi(error[0])
            self.assertTrue(np.allclose(b[2*(len(self.odoms)+1+i):2*(len(self.odoms)+2+i)], error, atol=1e-6))

if __name__ == '__main__':
    unittest.main()