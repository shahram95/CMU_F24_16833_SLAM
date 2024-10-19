import numpy as np
from scipy.sparse import csr_matrix
import unittest
from solvers import solve_pinv

class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.A = csr_matrix([[1, 2], [3, 4], [5, 6]])
        self.b = np.array([8, 18, 28])
    
    def test_solve_pinv(self):
        x, _ = solve_pinv(self.A, self.b)
        result = np.allclose(self.A @ x, self.b, atol=1e-6)
        print(f"solve_pinv test {'passed' if result else 'failed'}")
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main(verbosity=2)