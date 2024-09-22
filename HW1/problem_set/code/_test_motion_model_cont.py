# Add relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from motion_model import MotionModel

def generate_circular_path(center, radius, num_points):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack((x, y))

def generate_straight_line_path(start, end, num_points):
    t = np.linspace(0, 1, num_points)
    x = start[0] + t * (end[0] - start[0])
    y = start[1] + t * (end[1] - start[1])
    return np.column_stack((x,y))

def generate_figure_eight_path(center, size, num_points):
    t = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + size * np.sin(t)
    y = center[1] + size/2 * np.sin(t) * np.cos(t)
    return np.column_stack((x,y))

def generate_square_path(center, side_length, num_points):
    points_per_side = num_points // 4
    side = np.linspace(-side_length/2, side_length/2, points_per_side)
    top = np.column_stack((side, np.full(points_per_side, side_length/2)))
    right = np.column_stack((np.full(points_per_side, side_length/2), side[::-1]))
    bottom = np.column_stack((side[::-1], np.full(points_per_side, -side_length/2)))
    left = np.column_stack((np.full(points_per_side, -side_length/2), side))
    square = np.vstack((top, right, bottom, left))
    return square + center
    

def calculate_heading(path):
    dx = np.diff(path[:, 0])
    dy = np.diff(path[:, 1])
    return np.arctan2(dy, dx)

def test_motion_model(path_func, path_args, path_name):
    # Generate path
    path = path_func(*path_args)

    # Calculate heading
    heading = calculate_heading(path)
    heading = np.insert(heading, 0, heading[0])

    # Create full state array (x, y, theta)
    true_states = np.column_stack((path, heading))

    # Initialize Motion Model
    model = MotionModel()

    # Apply Motion Model
    estimated_states = []
    current_state = true_states[0]

    for i in range(1, len(true_states)):
        u_t0 = true_states[i-1]
        u_t1 = true_states[i]
        x_t1 = model.update(u_t0, u_t1, current_state)
        estimated_states.append(x_t1)
        current_state = x_t1
    
    estimated_states = np.array(estimated_states)

    # Analyze results

    true_path = true_states[1:]
    position_error = np.linalg.norm(estimated_states[:, :2] - true_path[:, :2], axis=1)
    heading_error = np.abs(estimated_states[:, 2] - true_path[:, 2])

    avg_position_error = np.mean(position_error)
    max_position_error = np.max(position_error)
    avg_heading_error = np.mean(heading error)
    max_heading_error = np.max(heading_error)

    print(f"\nResults for {path_name}:")
    print(f"Average position error: {avg_position_error:.4f}")
    print(f"Maximum position error: {max_position_error:.4f}")
    print(f"Average heading error: {avg_heading_error:.4f}")
    print(f"Maximum heading error: {max_heading_error:.4f}")

    # Visualize results

    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.plot(true_path[:, 0], true_path[:, 1], '-b', label='True Path')
    plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'r--', label='Estimated Path')
    plt.legend()
    plt.title(f'{path_name}: True vs Estimated Path')
    plt.axis('equal')

    plt.subplot(122)
    plt.plot(position_error, label='Position Error')
    plt.plot(heading_error, label='Heading Error')
    plt.legend()
    plt.title("Error over Time")

    plt.tight_layout()
    plt.show()


def test_gt_plot(path_func, path_args, path_name):
    path = path_func(*path_args)

    plt.figure(figsize=(8,8))
    plt.plot(path[:, 0], path[:, 1], 'b-')
    plt.scatter(path[0, 0], path[0, 1], color='g', s=100, label='Start')
    plt.scatter(path[-1, 0], path[-1, 1], color='r', s=100, label='End')
    plt.title(f'Ground Truth: {path_name}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    num_points = 100

    #test_gt_plot(generate_circular_path, ([0, 0], 5, num_points), "Circular Path")
    #test_gt_plot(generate_straight_line_path, ([-5, -5], [5, 5], num_points), "Straight Line Path")
    #test_gt_plot(generate_figure_eight_path, ([0, 0], 5, num_points), "Figure-Eight Path")
    #test_gt_plot(generate_square_path, ([0, 0], 10, num_points), "Square Path")

    test_motion_model(generate_circular_path, ([0, 0], 5, num_points), "Circular Path")
    test_motion_model(generate_straight_line_path, ([-5, -5], [5, 5], num_points), "Straight Line Path")
    test_motion_model(generate_figure_eight_path, ([0, 0], 5, num_points), "Figure-Eight Path")
    test_motion_model(generate_square_path, ([0, 0], 10, num_points), "Square Path")
