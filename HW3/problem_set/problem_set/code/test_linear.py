from linear import create_linear_system
import numpy as np

def test_create_linear_system():
    # Test inputs
    odoms = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    observations = np.array([[0, 0, 2.0, 2.0], [1, 0, 3.0, 3.0], [2, 1, 2.0, 4.0]])
    sigma_odom = np.eye(2)
    sigma_observation = np.eye(2)
    n_poses = 4
    n_landmarks = 2

    # Call the function
    A, b = create_linear_system(odoms, observations, sigma_odom, sigma_observation, n_poses, n_landmarks)

    # Convert sparse matrix A to dense for easier checking
    A_dense = A.toarray()

    # Check dimensions
    assert A_dense.shape == (14, 12), f"Expected A shape (14, 12), got {A_dense.shape}"
    assert b.shape == (14,), f"Expected b shape (14,), got {b.shape}"

    # Check prior on first pose
    assert np.allclose(A_dense[0:2, 0:2], np.eye(2)), "Prior on first pose is incorrect"

    # Check odometry constraints
    for i in range(3):
        assert np.allclose(A_dense[2*i+2:2*i+4, 2*i:2*i+4], np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])), f"Odometry constraint {i} is incorrect"
        assert np.allclose(b[2*i+2:2*i+4], odoms[i]), f"Odometry measurement {i} is incorrect"

    # Check landmark observations
    for i, obs in enumerate(observations):
        pose_idx, landmark_idx = obs[0:2].astype(int)
        assert np.allclose(A_dense[8+2*i:10+2*i, 2*pose_idx:2*pose_idx+2], np.array([[-1, 0], [0, -1]])), f"Landmark observation {i} (pose part) is incorrect"
        assert np.allclose(A_dense[8+2*i:10+2*i, 8+2*landmark_idx:10+2*landmark_idx], np.array([[1, 0], [0, 1]])), f"Landmark observation {i} (landmark part) is incorrect"
        assert np.allclose(b[8+2*i:10+2*i], obs[2:4]), f"Landmark measurement {i} is incorrect"

    print("All tests passed!")

# Run the test
test_create_linear_system()