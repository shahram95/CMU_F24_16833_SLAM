import numpy as np
import matplotlib.pyplot as plt
from motion_model import MotionModel

def run_test_case(model, u_t0, u_t1, x_t0, num_runs=1000):
    '''
    Run a single test case multiple times to gather statistical data.

    Args:
        model (MotionModel): The motion model to test
        u_t0 (np.array): Initial odometry reading [x, y, theta]
        u_t1 (np.array): Final odometry reading [x, y, theta]
        x_t0 (np.array): Initial state belief [x, y, theta]
        num_runs (int): Number of time to run the test
    '''
    return np.array([model.update(u_t0, u_t1, x_t0) for _ in range(num_runs)])

def visualize_results(results, expected, title):
    '''
    Visualize the distribution of results for a test case.

    Args:
        results (np.array): Array of results from multiple run
        expected (np.array): Expected results [x, y, theta]
        title (str): Title for the plot
    '''
    pass

def test_motion_model():
    '''
    Test the motion model with various scenarios

    This function defines the test cases to evaluate the performance of the motion 
    model under different movement conditions
    '''
    model = MotionModel()
    num_runs = 1000

    # Test case definitetion [To-Do: Incorporate code to read in json format]
    test_cases = [
        {
            'name' : 'Moving forward',
            'u_t0' : np.array([0, 0, 0]),
            'u_t1' : np.array([1, 0, 0]),
            'x_t0' : np.array([0, 0, 0]),
            'expected' : np.array([1, 0, 0])
        },
        {
            'name' : 'Rotating in place',
            'u_t0' : np.array([0, 0, 0]),
            'u_t1' : np.array([0, 0, np.pi/2]),
            'x_t0' : np.array([1, 1, 0]),
            'expected' : np.array([1, 1, np.pi/2])
        },
        {
            'name' : 'Moving diagonally',
            'u_t0' : np.array([0, 0, 0]),
            'u_t1' : np.array([1, 1, np.pi/4]),
            'x_t0' : np.array([0, 0, 0]),
            'expected' : np.array([np.sqrt(2), np.sqrt(2), np.pi/4])
        },
        {
            'name' : 'Complex Motion',
            'u_t0' : np.array([1, 1, np.pi/4]),
            'u_t1' : np.array([2, 3, np.pi/2]),
            'x_t0' : np.array([0, 0, 0]),
            'expected' : np.array([2, 3, np.pi/2])
        }
    ]

    for case in test_cases:
        print(f"\nTest Case: {case['name']}")
        print(f"Input: u_t0 = {case['u_t0']}, u_t1 = {case['u_t1']}, x_t0 = {case['x_t0']}")
        print(f"Expected: {case['expected']}")

        # Run the test case multiple times to gather data
        results = run_test_case(model, case['u_t0'], case['u_t1'], case['x_t0'], num_runs)
        mean, std = np.mean(results, axis=0), np.std(results, axis=0)

        print(f"Mean output: {mean}")
        print(f"Standard deviation: {std}")

        # Calculate absolute error
        error = np.abs(mean - case['expected'])
        print(f"absolute error: {error}")

        # Visualize
        visualize_results(results, case['expected'], case['name'])

if __name__ == "__main__":
    test_motion_model()