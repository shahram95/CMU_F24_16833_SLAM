'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi
    if angle_rad <= -np.pi:
        angle_rad += 2 * np.pi
    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2

    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))

    x_t, y_t, theta_t = init_pose.flatten()
    Sigma_t = init_pose_cov

    beta_is = init_measure[0::2, 0]
    r_is = init_measure[1::2, 0]

    angle_is = theta_t + beta_is

    l_xs = x_t + r_is * np.cos(angle_is)
    l_ys = y_t + r_is * np.sin(angle_is)

    landmark[0::2, 0] = l_xs
    landmark[1::2, 0] = l_ys

    for i in range(k):
        beta_i = beta_is[i]
        r_i = r_is[i]
        angle_i = angle_is[i]

        H_i = np.array([
            [1, 0, -r_i * np.sin(angle_i)],
            [0, 1,  r_i * np.cos(angle_i)]
        ])

        F_i = np.array([
            [-r_i * np.sin(angle_i), np.cos(angle_i)],
            [ r_i * np.cos(angle_i), np.sin(angle_i)]
        ])

        P_li = F_i @ init_measure_cov @ F_i.T

        idx = 2 * i
        landmark_cov[idx: idx + 2, idx: idx + 2] = P_li

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    d_t = control[0, 0]
    alpha_t = control[1, 0]

    x_t = X[0, 0]
    y_t = X[1, 0]
    theta_t = X[2, 0]

    X_pre = X.copy()
    X_pre[0, 0] = x_t + d_t * np.cos(theta_t)
    X_pre[1, 0] = y_t + d_t * np.sin(theta_t)
    X_pre[2, 0] = warp2pi(theta_t + alpha_t)

    F_t = np.eye(3 + 2 * k)
    F_t[0, 2] = -d_t * np.sin(theta_t)
    F_t[1, 2] = d_t * np.cos(theta_t)

    Q_t = np.zeros((3 + 2 * k, 3 + 2 * k))
    G_t = np.array([
        [np.cos(theta_t), -np.sin(theta_t), 0],
        [np.sin(theta_t),  np.cos(theta_t), 0],
        [0, 0, 1]
    ])

    Q_t[:3, :3] = G_t @ control_cov @ G_t.T

    P_pre = F_t @ P @ F_t.T + Q_t

    return X_pre, P_pre


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    robot_pose = X_pre[:3, 0]
    
    meas_jacobian = np.zeros((2 * k, 3 + 2 * k))
    noise_matrix = np.zeros((2 * k, 2 * k))
    expected_meas = np.zeros((2 * k, 1))

    for lm_idx in range(k):
        state_offset = 3 + 2 * lm_idx

        lm_position = X_pre[state_offset:state_offset+2, 0]

        rel_pos = lm_position - robot_pose[:2]
        sq_dist = np.sum(rel_pos**2)
        dist = np.sqrt(sq_dist)

        angle_exp = warp2pi(np.arctan2(rel_pos[1], rel_pos[0]) - robot_pose[2])
        range_exp = dist
        expected_meas[2 * lm_idx:2 * lm_idx + 2, 0] = [angle_exp, range_exp]

        lm_jacobian = np.zeros((2, 3 + 2 * k))

        lm_jacobian[0, :3] = [rel_pos[1] / sq_dist, -rel_pos[0] / sq_dist, -1]
        lm_jacobian[1, :3] = [-rel_pos[0] / dist, -rel_pos[1] / dist, 0]

        lm_jacobian[0, state_offset:state_offset+2] = [-rel_pos[1] / sq_dist, rel_pos[0] / sq_dist]
        lm_jacobian[1, state_offset:state_offset+2] = [rel_pos[0] / dist, rel_pos[1] / dist]

        meas_jacobian[2 * lm_idx:2 * lm_idx + 2, :] = lm_jacobian
        noise_matrix[2 * lm_idx:2 * lm_idx + 2, 2 * lm_idx:2 * lm_idx + 2] = measure_cov

    meas_residual = measure - expected_meas

    for lm_idx in range(k):
        meas_residual[2 * lm_idx, 0] = warp2pi(meas_residual[2 * lm_idx, 0])

    residual_cov = meas_jacobian @ P_pre @ meas_jacobian.T + noise_matrix
    kalman_gain = P_pre @ meas_jacobian.T @ np.linalg.inv(residual_cov)
    X_updated = X_pre + kalman_gain @ meas_residual
    P_updated = (np.eye(3 + 2 * k) - kalman_gain @ meas_jacobian) @ P_pre

    return X_updated, P_updated


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    plt.scatter(l_true[0::2], l_true[1::2], label='True Landmarks')
    plt.scatter(X[3::2], X[4::2], color='red', marker='x', s=100, label='Estimated Landmarks')
    plt.draw()
    plt.waitforbuttonpress(0)

    # Added

    euclidean_distances = np.zeros(k)
    mahalanobis_distances = np.zeros(k)

    indices = np.arange(k) * 2
    x_indices = 3 + indices
    y_indices = x_indices + 1

    dx = l_true[indices] - X[x_indices, 0]
    dy = l_true[indices + 1] - X[y_indices, 0]
    euclidean_distances = np.sqrt(dx**2 + dy**2)

    for i in range(k):
        difference_vector = np.array([dx[i], dy[i]])
        covariance_matrix = P[x_indices[i]:y_indices[i]+1, x_indices[i]:y_indices[i]+1]
        mahalanobis_distances[i] = np.sqrt(difference_vector @ covariance_matrix @ difference_vector.T)

        print(f"the Euclidean distance of landmark {i + 1} is {euclidean_distances[i]}")
        print(f"the Mahalanobis distance of landmark {i + 1} is {mahalanobis_distances[i]}")


def main():
    # TEST: Setup uncertainty parameters
    # sig_x = 0.25
    # sig_y = 0.1
    # sig_alpha = 0.1
    # sig_beta = 0.01
    # sig_r = 0.08

    # Exp 1:
    # sig_x = 2.5
    # sig_y = 0.1
    # sig_alpha = 0.1
    # sig_beta = 0.01
    # sig_r = 0.08

    # Exp 2:
    # sig_x = 0.025
    # sig_y = 0.1
    # sig_alpha = 0.1
    # sig_beta = 0.01
    # sig_r = 0.08

    # Exp 3:
    # sig_x = 0.25
    # sig_y = 1.0
    # sig_alpha = 0.1
    # sig_beta = 0.01
    # sig_r = 0.08

    # # Exp 4:
    # sig_x = 0.25
    # sig_y = 0.01
    # sig_alpha = 0.1
    # sig_beta = 0.01
    # sig_r = 0.08

    # Exp 5:
    # sig_x = 0.25
    # sig_y = 0.1
    # sig_alpha = 1.0
    # sig_beta = 0.01
    # sig_r = 0.08

    # # Exp 6:
    # sig_x = 0.25
    # sig_y = 0.1
    # sig_alpha = 0.01
    # sig_beta = 0.01
    # sig_r = 0.08

    # Exp 7:
    # sig_x = 0.25
    # sig_y = 0.1
    # sig_alpha = 0.1
    # sig_beta = 0.1
    # sig_r = 0.08

    # # Exp 8:
    # sig_x = 0.25
    # sig_y = 0.1
    # sig_alpha = 0.1
    # sig_beta = 0.001
    # sig_r = 0.08

    # Exp 9:
    # sig_x = 0.25
    # sig_y = 0.1
    # sig_alpha = 0.1
    # sig_beta = 0.01
    # sig_r = 0.8

    # Exp 10:
    sig_x = 0.25
    sig_y = 0.1
    sig_alpha = 0.1
    sig_beta = 0.01
    sig_r = 0.008


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("../data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
