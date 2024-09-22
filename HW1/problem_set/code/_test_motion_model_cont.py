# Add relevant libraries
import numpy as np
import matplotlib.pyplot as plt

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
    bottom = np.column_stack((side[::-1]. np.full(points_per_side, -side_length/2)))
    left = np.column_stack((np.full(points_per_side, -side_length/2), side))
    square = np.vstack((top, right, bottom, left))
    return square + center
    

def calculate_heading(path):
    pass

def test_motion_model():
    pass

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
    test_gt_plot(generate_square_path, ([0, 0], 10, num_points), "Square Path")