'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
    Modified to handle kidnapped robot scenario with adaptive reinitialization
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.title(f"Timestep: {tstep}")
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-np.pi, np.pi, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    x0_vals = []
    y0_vals = []
    theta0_vals = []

    map_height, map_width = occupancy_map.shape

    while len(x0_vals) < num_particles:
        y0_rand = np.random.uniform(0, map_height * 10, (num_particles, 1))
        x0_rand = np.random.uniform(0, map_width * 10, (num_particles, 1))
        theta_rand = np.random.uniform(-np.pi, np.pi, (num_particles, 1))
        x_map = np.round(x0_rand / 10.0).astype(np.int64)
        y_map = np.round(y0_rand / 10.0).astype(np.int64)
        for i in range(len(x_map)):
            if x_map[i] >= 0 and x_map[i] < occupancy_map.shape[1] and y_map[i] >= 0 and y_map[i] < occupancy_map.shape[0]:
                if np.abs(occupancy_map[y_map[i], x_map[i]]) == 0:
                    if len(x0_vals) < num_particles:
                        x0_vals.append(x0_rand[i, 0])
                        y0_vals.append(y0_rand[i, 0])
                        theta0_vals.append(theta_rand[i, 0])

    x0_vals = np.array(x0_vals).reshape(-1, 1)
    y0_vals = np.array(y0_vals).reshape(-1, 1)
    theta0_vals = np.array(theta0_vals).reshape(-1, 1)

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))
    return X_bar_init


def init_particles_around_pose(num_particles, x, y, theta, std_dev, occupancy_map):
    """
    Initialize particles around a given pose with Gaussian noise.
    """
    w0_vals = np.ones((num_particles, 1), dtype=np.float64) / num_particles

    x0_vals = []
    y0_vals = []
    theta0_vals = []

    while len(x0_vals) < num_particles:
        x_samples = np.random.normal(x, std_dev, (num_particles, 1))
        y_samples = np.random.normal(y, std_dev, (num_particles, 1))
        theta_samples = np.random.normal(theta, std_dev / 10.0, (num_particles, 1))
        x_map = np.round(x_samples / 10.0).astype(np.int64)
        y_map = np.round(y_samples / 10.0).astype(np.int64)
        for i in range(len(x_map)):
            if x_map[i] >= 0 and x_map[i] < occupancy_map.shape[1] and y_map[i] >= 0 and y_map[i] < occupancy_map.shape[0]:
                if np.abs(occupancy_map[y_map[i], x_map[i]]) == 0:
                    if len(x0_vals) < num_particles:
                        x0_vals.append(x_samples[i, 0])
                        y0_vals.append(y_samples[i, 0])
                        theta0_vals.append(theta_samples[i, 0])

    x0_vals = np.array(x0_vals).reshape(-1, 1)
    y0_vals = np.array(y0_vals).reshape(-1, 1)
    theta0_vals = np.array(theta0_vals).reshape(-1, 1)

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))
    return X_bar_init


if __name__ == '__main__':
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='data/map/wean.dat')
    parser.add_argument('--path_to_log', default='data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_false')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)

    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True

    # Variables for kidnapping detection
    kidnapped = False
    kidnapping_counter = 0
    motion_model_normal_params = (motion_model._alpha1, motion_model._alpha2, motion_model._alpha3, motion_model._alpha4)
    motion_model_kidnapped_params = (0.1, 0.1, 0.2, 0.2)  # Increased noise parameters
    kidnapping_duration = 20  # Number of timesteps to keep increased noise

    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # Convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # Odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        if meas_type == "L":
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        for m in range(num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            """
            SENSOR MODEL
            """
            if meas_type == "L":
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        # Normalize weights
        weights = X_bar[:, 3]
        weights += 1e-300  # Prevent division by zero
        weights /= np.sum(weights)
        X_bar[:, 3] = weights

        # Compute maximum weight
        max_weight = np.max(weights)

        # Threshold for kidnapping detection
        max_weight_threshold = 1e-4  # Adjust this threshold based on experiments
        print(max_weight)

        if max_weight < max_weight_threshold:
            # Kidnapping detected
            kidnapped = True
            kidnapping_counter = 0
            print("Kidnapping detected at timestep {}".format(time_idx))

            # Increase motion model noise parameters
            motion_model._alpha1, motion_model._alpha2, motion_model._alpha3, motion_model._alpha4 = motion_model_kidnapped_params

            # Hybrid Reinitialization
            num_particles_near = num_particles // 2
            num_particles_random = num_particles - num_particles_near

            # Estimate last known position (weighted mean)
            x_est = np.sum(X_bar[:, 0] * weights)
            y_est = np.sum(X_bar[:, 1] * weights)
            theta_est = np.arctan2(np.sum(np.sin(X_bar[:, 2]) * weights), np.sum(np.cos(X_bar[:, 2]) * weights))

            # Initialize particles around last known position
            X_near = init_particles_around_pose(num_particles_near, x_est, y_est, theta_est, std_dev=500, occupancy_map=occupancy_map)

            # Initialize particles randomly across the map
            X_random = init_particles_freespace(num_particles_random, occupancy_map)

            # Combine particles
            X_bar = np.vstack((X_near, X_random))

            # Reset weights
            X_bar[:, 3] = 1.0 / num_particles

        if kidnapped:
            kidnapping_counter += 1
            if kidnapping_counter > kidnapping_duration:
                # Reset motion model noise parameters to normal
                motion_model._alpha1, motion_model._alpha2, motion_model._alpha3, motion_model._alpha4 = motion_model_normal_params
                kidnapped = False
                print("Ending increased noise period at timestep {}".format(time_idx))

        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)
