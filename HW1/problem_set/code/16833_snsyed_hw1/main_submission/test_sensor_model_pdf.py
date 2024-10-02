import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def p_hit(z, z_star, sigma_hit, z_max):
    if 0 <= z <= z_max:
        eta = 1 / (norm.cdf(z_max, loc=z_star, scale=sigma_hit) - norm.cdf(0, loc=z_star, scale=sigma_hit))
        return eta * norm.pdf(z, loc=z_star, scale=sigma_hit)
    else:
        return 0.0

def p_short(z, z_star, lambda_short):
    if 0 <= z <= z_star:
        eta = 1 / (1 - np.exp(-lambda_short * z_star))
        return eta * lambda_short * np.exp(-lambda_short * z)
    else:
        return 0.0

def p_max(z, z_max):
    if z >= z_max - 1.0:  # Small threshold for numerical stability
        return 1.0
    else:
        return 0.0

def p_rand(z, z_max):
    if 0 <= z < z_max:
        return 1.0 / z_max
    else:
        return 0.0

def test_sensor_model_distribution_separate_plots():
    # Sensor model parameters
    w_hit = 0.7
    w_short = 0.1
    w_max = 0.1
    w_rand = 0.1

    sigma_hit = 100.0      # Standard deviation for p_hit
    lambda_short = 0.1    # Parameter for p_short
    z_max_range = 2500     # Max range of the sensor in mm

    # Expected measurement z_star
    z_star_values = [500, 2000, 4000]  # Test different expected measurements

    for z_star in z_star_values:
        # Generate range of z values
        z_values = np.linspace(0, z_max_range, 1000)

        # Compute p(z | z_star) for each z
        p_hit_values = np.array([p_hit(z, z_star, sigma_hit, z_max_range) for z in z_values])
        p_short_values = np.array([p_short(z, z_star, lambda_short) for z in z_values])
        p_max_values = np.array([p_max(z, z_max_range) for z in z_values])
        p_rand_values = np.array([p_rand(z, z_max_range) for z in z_values])

        # Apply weights
        p_hit_values *= w_hit
        p_short_values *= w_short
        p_max_values *= w_max
        p_rand_values *= w_rand

        # Combined probability
        p_values = p_hit_values + p_short_values + p_max_values + p_rand_values

        # Create separate plots for each component
        plt.figure(figsize=(10, 8))

        # p_hit
        plt.subplot(5, 1, 1)
        plt.plot(z_values, p_hit_values, color='blue')
        plt.title(f'p_hit Component (z_star = {z_star} mm)')
        plt.ylabel('Probability')
        plt.grid(True)

        # p_short
        plt.subplot(5, 1, 2)
        plt.plot(z_values, p_short_values, color='green')
        plt.title('p_short Component')
        plt.ylabel('Probability')
        plt.grid(True)

        # p_max
        plt.subplot(5, 1, 3)
        plt.plot(z_values, p_max_values, color='red')
        plt.title('p_max Component')
        plt.ylabel('Probability')
        plt.grid(True)

        # p_rand
        plt.subplot(5, 1, 4)
        plt.plot(z_values, p_rand_values, color='purple')
        plt.title('p_rand Component')
        plt.ylabel('Probability')
        plt.grid(True)

        # Combined p(z | z*)
        plt.subplot(5, 1, 5)
        plt.plot(z_values, p_values, color='black')
        plt.title('Combined Sensor Model p(z | z*)')
        plt.xlabel('Measured range z (mm)')
        plt.ylabel('Probability')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_sensor_model_distribution_separate_plots()
