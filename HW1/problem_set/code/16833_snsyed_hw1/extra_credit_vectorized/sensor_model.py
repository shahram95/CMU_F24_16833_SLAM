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
from tqdm import tqdm
import multiprocessing as mp
import os

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
        self._z_hit = 5
        self._z_short = 0.15
        self._z_max = 0.05
        self._z_rand = 1000

        self._sigma_hit = 150
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

        self.angle_step = 1  # degrees
        self.position_step = 10  # cm
        # print("Precomputing ray cast table...")
        # self.ray_cast_table = self.precompute_ray_cast_table()
        # print("Precomputation complete.")

        self.ray_cast_table_filename = 'ray_cast_table.npy'
        
        print("Checking for precomputed ray cast table...")
        if os.path.exists(self.ray_cast_table_filename):
            self.load_ray_cast_table()
        else:
            self.compute_and_save_ray_cast_table()
        
    def load_ray_cast_table(self):
        print(f"Loading ray cast table from {self.ray_cast_table_filename}")
        self.ray_cast_table = np.load(self.ray_cast_table_filename)
        print("Ray cast table loaded successfully.")

    def compute_and_save_ray_cast_table(self):
        print("Precomputing ray cast table...")
        self.ray_cast_table = self.precompute_ray_cast_table()
        print("Precomputation complete.")
        
        print(f"Saving ray cast table to {self.ray_cast_table_filename}")
        np.save(self.ray_cast_table_filename, self.ray_cast_table)
        print("Ray cast table saved successfully.")
    
    
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
    
    def precompute_ray_cast_table(self):
        print("Precomputing ray cast table...")
        x_steps = self.map_width // (self.position_step // self.resolution)
        y_steps = self.map_height // (self.position_step // self.resolution)
        angle_steps = 360 // self.angle_step

        table = np.zeros((y_steps, x_steps, angle_steps))

        for y in tqdm(range(y_steps)):
            for x in range(x_steps):
                for a in range(angle_steps):
                    start_x = x * self.position_step
                    start_y = y * self.position_step
                    angle = math.radians(a * self.angle_step)
                    table[y, x, a] = self.ray_casting(start_x, start_y, angle)

        return table
    
    def lookup_ray_cast(self, start_x, start_y, angle):
        x_idx = int(start_x / self.position_step)
        y_idx = int(start_y / self.position_step)
        angle_idx = int(math.degrees(angle) % 360) // self.angle_step

        x_idx = min(x_idx, self.ray_cast_table.shape[1] - 1)
        y_idx = min(y_idx, self.ray_cast_table.shape[0] - 1)

        return self.ray_cast_table[y_idx, x_idx, angle_idx]

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

    
    def lookup_ray_cast_vectorized(self, start_x, start_y, angle):
        """
        Vectorized lookup for expected ranges using the precomputed ray cast table.
        
        Args:
            start_x (np.ndarray): Array of x positions of the laser (flattened).
            start_y (np.ndarray): Array of y positions of the laser (flattened).
            angle (np.ndarray): Array of beam angles (flattened).
        
        Returns:
            np.ndarray: Array of expected ranges corresponding to each (start_x, start_y, angle).
        """
        x_idx = (start_x / self.position_step).astype(np.int32)
        y_idx = (start_y / self.position_step).astype(np.int32)
        angle_idx = (np.degrees(angle) % 360 // self.angle_step).astype(np.int32)
        
        # Ensure indices are within bounds
        x_idx = np.clip(x_idx, 0, self.ray_cast_table.shape[1] - 1)
        y_idx = np.clip(y_idx, 0, self.ray_cast_table.shape[0] - 1)
        angle_idx = np.clip(angle_idx, 0, self.ray_cast_table.shape[2] - 1)
        
        # Fetch expected ranges from the precomputed table
        z_star_flat = self.ray_cast_table[y_idx, x_idx, angle_idx]
        return z_star_flat
    
    def beam_range_finder_model(self, z_t1_arr, X_t1):
        """
        Vectorized calculation of the likelihood of a range scan at time t for all particles.
        """
        num_particles = X_t1.shape[0]
        num_beams = len(range(0, 180, self._subsampling))

        x = X_t1[:, 0]
        y = X_t1[:, 1]
        theta = X_t1[:, 2]

        # Compute laser positions
        laser_x = x + self.offset * np.cos(theta)
        laser_y = y + self.offset * np.sin(theta)

        # Prepare beam angles
        beam_indices = np.arange(0, 180, self._subsampling)
        beam_angles = np.radians(beam_indices - 90)  # Shape: (num_beams,)
        beam_angles = beam_angles.reshape(1, -1) + theta.reshape(-1, 1)  # Shape: (num_particles, num_beams)

        # Flatten arrays for vectorized ray casting
        laser_x_flat = np.repeat(laser_x, num_beams)
        laser_y_flat = np.repeat(laser_y, num_beams)
        beam_angles_flat = beam_angles.flatten()

        # Expected ranges
        z_star_flat = self.lookup_ray_cast_vectorized(laser_x_flat, laser_y_flat, beam_angles_flat)
        z_star = z_star_flat.reshape(num_particles, num_beams)

        # Measured ranges
        z = z_t1_arr[::self._subsampling]
        z = np.tile(z, (num_particles, 1))

        # Compute probabilities
        p_hit = self._z_hit * norm.pdf(z, z_star, self._sigma_hit)
        p_short = np.where((z <= z_star) & (z >= 0),
                           self._z_short * self._lambda_short * np.exp(-self._lambda_short * z), 0)
        p_max = np.where(z >= self._max_range - 5, self._z_max, 0)
        p_rand = np.where((z >= 0) & (z < self._max_range), self._z_rand / self._max_range, 0)

        p = p_hit + p_short + p_max + p_rand
        p = np.maximum(p, 1e-300)  # Avoid log(0)

        # Compute log probabilities and sum over beams
        log_p = np.log(p)
        log_prob = np.sum(log_p, axis=1)

        # Return weights
        weights = np.exp(log_prob)
        return weights
