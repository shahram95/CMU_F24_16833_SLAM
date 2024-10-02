'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        # Noen neede for this method

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        weights = X_bar[:, 3]

        # Added check for negative weights and zero sum

        if np.any(weights < 0):
            raise ValueError("Weights cannot be negative")

        weight_sum = np.sum(weights)
        if weight_sum == 0:
            raise ValueError("Sum of weights cannot be zero")

        normalized_weights = weights / weight_sum

        resampled_indices = np.random.multinomial(num_particles, normalized_weights)

        X_bar_resampled =  np.repeat(X_bar, resampled_indices, axis=0)
        X_bar_resampled[:, 3] = 1.0 / num_particles

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        weights = X_bar[:, 3]

        # Check for negative weights and zero sum
        if np.any(weights < 0):
            raise ValueError("Weights cannot be negative")

        weight_sum = np.sum(weights)
        if weight_sum == 0:
            raise ValueError("Sum of weights cannot be zero")

        normalized_weights = weights / weight_sum

        X_bar_resampled = np.zeros_like(X_bar)

        start_point = np.random.uniform(0, 1.0 / num_particles)
        cumulative_weight = normalized_weights[0]
        index = 0

        for particle_idx in range(num_particles):
            threshold = start_point + particle_idx / num_particles
            while threshold > cumulative_weight:
                index += 1
                cumulative_weight += normalized_weights[index]
            X_bar_resampled[particle_idx] = X_bar[index]
        
        X_bar_resampled[:, 3] = 1.0 / num_particles

        return X_bar_resampled
