import numpy as np
import matplotlib.pyplot as plt
from map_reader import MapReader
from motion_model import MotionModel

def test_motion_model_trajectory(map_path, log_path):
    # Load the occupancy map
    map_reader = MapReader(map_path)
    occupancy_map = map_reader.get_map()

    # Initialize the motion model
    motion_model = MotionModel()

    # Initialize lists to store the particle's trajectory
    trajectory = []

    # Open the log file and read the odometry data
    with open(log_path, 'r') as logfile:
        lines = logfile.readlines()

    # Initialize the particle's state using the first odometry reading
    first_line_found = False
    for line in lines:
        meas_type = line[0]
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        if meas_type == 'O' or meas_type == 'L':
            # Odometry readings
            u_t0 = meas_vals[0:3]
            x_t0 = u_t0.copy()
            trajectory.append(x_t0)
            first_line_found = True
            break

    if not first_line_found:
        print("No odometry data found in log file.")
        return

    # Process the log file to update the particle's state over time
    for line in lines:
        meas_type = line[0]
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        if meas_type == 'O' or meas_type == 'L':
            u_t1 = meas_vals[0:3]

            # Update particle state using the motion model
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            # Append the new state to the trajectory
            trajectory.append(x_t1)

            # Prepare for the next iteration
            u_t0 = u_t1
            x_t0 = x_t1

    trajectory = np.array(trajectory)

    # Plot the occupancy map and the particle's trajectory
    plt.figure(figsize=(10, 10))
    plt.imshow(occupancy_map, cmap='Greys', origin='lower')
    plt.plot(400+ trajectory[:, 1] / 10.0, 400 + trajectory[:, 0] / 10.0, 'r-', label='Particle Trajectory')
    plt.title('Particle Trajectory using Motion Model')
    plt.xlabel('X position (decimeters)')
    plt.ylabel('Y position (decimeters)')
    plt.legend()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    map_path = '../data/map/wean.dat'       # Update this path if necessary
    log_path = '../data/log/robotdata1.log' # Update this path if necessary
    test_motion_model_trajectory(map_path, log_path)
