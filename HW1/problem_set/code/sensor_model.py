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
        self._z_short = 0.12
        self._z_max = 0.05
        self._z_rand = 800

        self._sigma_hit = 100
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 8183

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        # Definitions
        self.map = occupancy_map
        self.map_size = occupancy_map.shape[0]
        self.map_height, self.map_width = occupancy_map.shape
        self.resolution = 10
        self.offset = 25
    
    def ray_cast_uv(self, start_x, start_y, angle):
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

    def beam_range_finder_model_uv(self, z_t1_arr, x_t1):
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
            prob_zt1 *= beam_prob if beam_prob > 0 else 1e-300

        return prob_zt1

    def ray_casting(self, start_x, start_y, angle):
        x_map, y_map = int(start_x / self.resolution), int(start_y / self.resolution)

        if not self._is_in_map(x_map, y_map):
            return self._max_range

        cos_theta, sin_theta = np.cos(angle), np.sin(angle)
        for r in range(0, self._max_range, 10):
            end_x = start_x + r * cos_theta
            end_y = start_y + r * sin_theta

            x_map_end, y_map_end = int(end_x / self.resolution), int(end_y / self.resolution)

            if not self._is_in_map(x_map_end, y_map_end):
                return r
            
            if self.map[y_map_end, x_map_end] > self._min_probability:
                return r
        
        return self._max_range
    
    def _is_in_map(self,x,y):
        return 0 <= x < self.map_width and 0 <= y < self.map_height
    
    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        Calculate the likelihood of a range scan at time t.
        
        Args:
            z_t1_arr : laser range readings [array of 180 values] at time t
            x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        
        Returns:
            float: Likelihood of a range scan zt1 at time t
        """
        x, y, theta = x_t1
        
        laser_x = x + self.offset * np.cos(theta)
        laser_y = y + self.offset * np.sin(theta)
        
        log_prob = 0.0
        
        for i in range(0, 180, self._subsampling):
            z = z_t1_arr[i]
            angle = theta + np.radians(i - 90)
            z_star = self.ray_casting(laser_x, laser_y, angle)
            
            p_hit = self._z_hit * norm.pdf(z, z_star, self._sigma_hit)
            p_short = self._z_short * self._lambda_short * np.exp(-self._lambda_short * z) if z <= z_star else 0
            p_max = self._z_max if z >= self._max_range else 0
            p_rand = self._z_rand / self._max_range if z < self._max_range else 0
            
            p = p_hit + p_short + p_max + p_rand
            
            log_prob += np.log(max(p, 1e-300))  # Avoid log(0)
        
        return np.exp(log_prob)