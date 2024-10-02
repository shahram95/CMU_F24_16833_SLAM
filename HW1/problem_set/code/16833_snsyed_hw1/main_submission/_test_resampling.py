import numpy as np
import unittest
from resampling import Resampling

class TestResampling(unittest.TestCase):
    def setUp(self):
        self.resampling = Resampling()

    def test_multinomial_sampler_output_shape(self):
        X_bar = np.array([[1, 2, 3, 0.1], [4, 5, 6, 0.2], [7, 8, 9, 0.7]])
        result = self.resampling.multinomial_sampler(X_bar)
        self.assertEqual(result.shape, X_bar.shape)

    def test_low_variance_sampler_output_shape(self):
        X_bar = np.array([[1, 2, 3, 0.1], [4, 5, 6, 0.2], [7, 8, 9, 0.7]])
        result = self.resampling.low_variance_sampler(X_bar)
        self.assertEqual(result.shape, X_bar.shape)

    def test_multinomial_sampler_weight_sum(self):
        X_bar = np.array([[1, 2, 3, 0.1], [4, 5, 6, 0.2], [7, 8, 9, 0.7]])
        result = self.resampling.multinomial_sampler(X_bar)
        self.assertAlmostEqual(np.sum(result[:, 3]), 1.0, places=6)

    def test_low_variance_sampler_weight_sum(self):
        X_bar = np.array([[1, 2, 3, 0.1], [4, 5, 6, 0.2], [7, 8, 9, 0.7]])
        result = self.resampling.low_variance_sampler(X_bar)
        self.assertAlmostEqual(np.sum(result[:, 3]), 1.0, places=6)

    def test_multinomial_sampler_zero_weights(self):
        X_bar = np.array([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]])
        with self.assertRaises(ValueError):
            self.resampling.multinomial_sampler(X_bar)

    def test_low_variance_sampler_zero_weights(self):
        X_bar = np.array([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]])
        with self.assertRaises(ValueError):
            self.resampling.low_variance_sampler(X_bar)

    def test_multinomial_sampler_negative_weights(self):
        X_bar = np.array([[1, 2, 3, -0.1], [4, 5, 6, 0.2], [7, 8, 9, 0.9]])
        with self.assertRaises(ValueError):
            self.resampling.multinomial_sampler(X_bar)

    def test_low_variance_sampler_negative_weights(self):
        X_bar = np.array([[1, 2, 3, -0.1], [4, 5, 6, 0.2], [7, 8, 9, 0.9]])
        with self.assertRaises(ValueError):
            self.resampling.low_variance_sampler(X_bar)

    def test_multinomial_sampler_single_particle(self):
        X_bar = np.array([[1, 2, 3, 1.0]])
        result = self.resampling.multinomial_sampler(X_bar)
        np.testing.assert_array_equal(result, X_bar)

    def test_low_variance_sampler_single_particle(self):
        X_bar = np.array([[1, 2, 3, 1.0]])
        result = self.resampling.low_variance_sampler(X_bar)
        np.testing.assert_array_equal(result, X_bar)

    def test_multinomial_sampler_consistency(self):
        X_bar = np.array([[1, 2, 3, 0.1], [4, 5, 6, 0.2], [7, 8, 9, 0.7]])
        np.random.seed(42)
        result1 = self.resampling.multinomial_sampler(X_bar)
        np.random.seed(42)
        result2 = self.resampling.multinomial_sampler(X_bar)
        np.testing.assert_array_equal(result1, result2)

    def test_low_variance_sampler_consistency(self):
        X_bar = np.array([[1, 2, 3, 0.1], [4, 5, 6, 0.2], [7, 8, 9, 0.7]])
        np.random.seed(42)
        result1 = self.resampling.low_variance_sampler(X_bar)
        np.random.seed(42)
        result2 = self.resampling.low_variance_sampler(X_bar)
        np.testing.assert_array_equal(result1, result2)

    def test_multinomial_sampler_large_input(self):
        X_bar = np.random.rand(10000, 4)
        X_bar[:, 3] /= np.sum(X_bar[:, 3])
        result = self.resampling.multinomial_sampler(X_bar)
        self.assertEqual(result.shape, X_bar.shape)

    def test_low_variance_sampler_large_input(self):
        X_bar = np.random.rand(10000, 4)
        X_bar[:, 3] /= np.sum(X_bar[:, 3])
        result = self.resampling.low_variance_sampler(X_bar)
        self.assertEqual(result.shape, X_bar.shape)

if __name__ == '__main__':
    unittest.main()