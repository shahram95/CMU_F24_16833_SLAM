import numpy as np
from scipy.sparse import csr_matrix
import unittest
from solvers import solve_pinv, solve_lu, solve_lu_colamd, solve_qr, solve_qr_colamd

class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.A = csr_matrix([[1, 2], [3, 4], [5, 6]])
        self.b = np.array([8, 18, 28])
    
    def test_solve_pinv(self):
        x, _ = solve_pinv(self.A, self.b)
        result = np.allclose(self.A @ x, self.b, atol=1e-6)
        print(f"solve_pinv test {'passed' if result else 'failed'}")
        self.assertTrue(result)
    
    def test_solve_lu(self):
        x, U = solve_lu(self.A, self.b)
        result1 = np.allclose(self.A @ x, self.b, atol=1e-6)
        result2 = U.shape[0] == U.shape[1]  # Check if U is square
        print(f"solve_lu test {'passed' if result1 and result2 else 'failed'}")
        self.assertTrue(result1 and result2)
    
    def test_solve_lu_colamd(self):
        x, U = solve_lu_colamd(self.A, self.b)
        result1 = np.allclose(self.A @ x, self.b, atol=1e-6)
        result2 = U.shape[0] == U.shape[1]  # Check if U is square
        print(f"solve_lu_colamd test {'passed' if result1 and result2 else 'failed'}")
        self.assertTrue(result1 and result2)
    
    def test_solve_qr(self):
        x, R = solve_qr(self.A, self.b)
        result1 = np.allclose(self.A @ x, self.b, atol=1e-6)
        result2 = R.shape[0] == R.shape[1]  # Check if R is square
        print(f"solve_qr test {'passed' if result1 and result2 else 'failed'}")
        self.assertTrue(result1 and result2)
    
    def test_solve_qr_colamd(self):
        x, R = solve_qr_colamd(self.A, self.b)
        result1 = np.allclose(self.A @ x, self.b, atol=1e-6)
        result2 = R.shape[0] == R.shape[1]  # Check if R is square
        print(f"solve_qr_colamd test {'passed' if result1 and result2 else 'failed'}")
        self.assertTrue(result1 and result2)

if __name__ == '__main__':
    unittest.main(verbosity=2)