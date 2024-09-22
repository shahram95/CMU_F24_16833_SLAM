'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


def occupancy(x, res, occ_map):
    x_idx = int(math.floor(x[0] / res ))
    y_idx = int(math.floor(x[1] / res ))
    return occ_map[y_idx, x_idx]

def check_bounds(coord, res, map_size):
    x_idx = math.floor(coord[0] / res)
    y_idx = math.floor(coord[1] / res)
    return (0 <= x_idx < map_size) and (0 <= y_idx < map_size)

class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        # Definitions
        self.map = occupancy_map
        self.map_size = occupancy_map.shape[0]
        self.resolution = 10
        self.offset = 25
    
    def ray_cast(self, start_x, start_y, angle):
        end_x = start_x
        end_y = start_y
        x_idx = int(round(end_x/self.resolution))
        y_idx = int(round(end_y/self.resolution))

        cell_value = occupancy([x_idx, y_idx], self.resolution, self.map)
        while check_bounds([x_idx, y_idx], self.resolution, self.map_size) and cell_value < self._min_probability:
            cell_value = occupancy([x_idx, y_idx], self.resolution, self.map)
            end_x += 8 * math.cos(angle)
            end_y += 8 * math.sin(angle)
            x_idx = int(round(end_x/self.resolution))
            y_idx = int(round(end_y/self.resolution))
        
        distance = math.sqrt((start_x-end_x)**2 + (start_y-end_y)**2)
        return distance

    def calculate_probability(self, expected, measured):
        total_prob = 0

        if measured >= 0:
            hit_prob = self._z_hit * norm.pdf(measured, expected, self._sigma_hit) if 0 <= measured <= self._max_range else 0
            short_prob = self._z_short * self._lambda_short * math.exp(-self._lambda_short * measured) if 0 <= measured <= expected else 0
            max_prob = self._z_max if measured >= self._max_range -5 else 0
            rand_prob = self._z_rand / self._max_range if measured < self._max_range else 0
            total_prob = hit_prob + short_prob + max_prob + rand_prob
        
        return total_prob

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        prob_zt1 = 1.0
        x, y, theta = x_t1
        laser_x = x + self.offset * math.cos(theta)
        laser_y = y + self.offset * math.sin(theta)

        for i in range(0, 180, self._subsampling):
            measured_range = z_t1_arr[i]
            beam_angle = theta + math.radians(i - 90)
            expected_range = self.ray_cast(laser_x, laser_y, beam_angle)
            beam_prob = self.calculate_probability(expected_range, measured_range)
            prob_zt1 *- beam_prob if beam_prob > 0 else 1e-300

        return prob_zt1
