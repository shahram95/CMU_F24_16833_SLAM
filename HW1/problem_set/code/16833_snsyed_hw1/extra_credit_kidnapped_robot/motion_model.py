'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.00001
        self._alpha2 = 0.00001
        self._alpha3 = 0.0001
        self._alpha4 = 0.0001
    
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        # calculate relative motion
        rot1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        trans = np.sqrt((u_t1[0] - u_t0[0])**2 + (u_t1[1] - u_t0[1])**2)
        rot2 = u_t1[2] - u_t0[2] - rot1

        # Normalize angle between -pi and pi to handle angle overflow
        rot1 = self.normalize_angle(rot1)
        rot2 = self.normalize_angle(rot2)

        # Add noise; sqrt because np.random.normal doesnot take variance instead require standard deviation
        noise_rot1 = np.sqrt(self._alpha1 * rot1**2 + self._alpha2 * trans**2)
        noise_trans = np.sqrt(self._alpha3 * trans**2 + self._alpha4 * (rot1**2 + rot2**2))
        noise_rot2 = np.sqrt(self._alpha1 * rot2**2 + self._alpha2 * trans**2)

        rot1_with_noise = rot1 - np.random.normal(0, noise_rot1)
        trans_with_noise = trans - np.random.normal(0, noise_trans)
        rot2_with_noise = rot2 - np.random.normal(0, noise_rot2)

        # Calculate new state
        x_t1 = np.zeros(3)
        x_t1[0] = x_t0[0] + trans_with_noise * np.cos(x_t0[2] + rot1_with_noise)
        x_t1[1] = x_t0[1] + trans_with_noise * np.sin(x_t0[2] + rot1_with_noise)
        x_t1[2] = self.normalize_angle(x_t0[2] + rot1_with_noise + rot2_with_noise)
        
        return x_t1

    def update_optimized(self, u_t0, u_t1, x_t0):
        '''
        Optimized update method using vectorized operations.

        Args:
            u_t0 (np.ndarray): Particle state odometry reading [x, y, theta] at time (t-1) [odometry frame]
            u_t1 (np.ndarray): Particle state odometry reading [x, y, theta] at time (t) [odometry frame]
            x_t0 (np.ndarray): Particle state belief [x, y, theta] at time (t-1) [world frame]
        
            Returns:
                np.ndarray: Updated particle state belief [x, y, theta] at time t [world frame]
        '''
        # calculate relative motion
        delta_x, delta_y = u_t1[:2] - u_t0[:2]
        rot1 = np.arctan2(delta_y, delta_x) - u_t0[2]
        trans = np.hypot(delta_x, delta_y)
        rot2 = u_t1[2] - u_t0[2] - rot1

        # Normalize angle between -pi and pi to handle angle overflow
        rot1, rot2 = map(self.normalize_angle, (rot1, rot2))

        # Add noise; sqrt because np.random.normal doesnot take variance instead require standard deviation
        noise_rot1 = np.sqrt(self._alpha1 * rot1**2 + self._alpha2 * trans**2)
        noise_trans = np.sqrt(self._alpha3 * trans**2 + self._alpha4 * (rot1**2 + rot2**2))
        noise_rot2 = np.sqrt(self._alpha1 * rot2**2 + self._alpha2 * trans**2)

        rot1_with_noise = rot1 - np.random.normal(0, noise_rot1)
        trans_with_noise = trans - np.random.normal(0, noise_trans)
        rot2_with_noise = rot2 - np.random.normal(0, noise_rot2)

        # Calculate new state
        cos_term = np.cos(x_t0[2] + rot1_with_noise)
        sin_term = np.sin(x_t0[2] + rot1_with_noise)

        x_t1 = np.array([
            x_t0[0] + trans_with_noise * cos_term,
            x_t0[1] + trans_with_noise * sin_term,
            self.normalize_angle(x_t0[2] + rot1_with_noise + rot2_with_noise)
        ])

        return x_t1
