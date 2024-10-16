import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

# Constants for the robot's motion
x_t = 2
y_t = 2
theta_t = np.pi / 4  # 45 degrees
d_t = 3
alpha_t = np.pi / 6  # 30 degrees

# Calculate new position
x_t1 = x_t + d_t * np.cos(theta_t)
y_t1 = y_t + d_t * np.sin(theta_t)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Robot Motion Model Visualization with Coordinate Axes')

# Plot initial and final positions
ax.plot([x_t, x_t1], [y_t, y_t1], 'ro-')  # Red 'o' for positions, '-' for path
ax.annotate('Initial Position (x_t, y_t)', (x_t, y_t), textcoords="offset points", xytext=(-10,-15), ha='center')
ax.annotate('New Position (x_t1, y_t1)', (x_t1, y_t1), textcoords="offset points", xytext=(-10,15), ha='center')

# Draw the robot's travel arrow
ax.arrow(x_t, y_t, d_t * np.cos(theta_t), d_t * np.sin(theta_t), head_width=0.2, head_length=0.2, fc='blue', ec='blue')

# Draw the orientation arrows
ax.arrow(x_t, y_t, 0.5 * np.cos(theta_t), 0.5 * np.sin(theta_t), head_width=0.1, head_length=0.1, fc='green', ec='green')
ax.arrow(x_t1, y_t1, 0.5 * np.cos(theta_t + alpha_t), 0.5 * np.sin(theta_t + alpha_t), head_width=0.1, head_length=0.1, fc='purple', ec='purple')

# Coordinate axes for each robot pose
# Initial position coordinate axes
ax.arrow(x_t, y_t, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black', label='X-axis at initial')
ax.arrow(x_t, y_t, 0, 1, head_width=0.1, head_length=0.1, fc='black', ec='black', label='Y-axis at initial')

# Final position coordinate axes (rotated by alpha_t)
ax.arrow(x_t1, y_t1, 1 * np.cos(alpha_t), 1 * np.sin(alpha_t), head_width=0.1, head_length=0.1, fc='orange', ec='orange', label='X-axis at final')
ax.arrow(x_t1, y_t1, -1 * np.sin(alpha_t), 1 * np.cos(alpha_t), head_width=0.1, head_length=0.1, fc='orange', ec='orange', label='Y-axis at final')

# Annotate angle alpha_t
angle_arc = patches.Arc((x_t1, y_t1), 1, 1, angle=np.degrees(theta_t), theta1=0, theta2=np.degrees(alpha_t), color='red')
ax.add_patch(angle_arc)
ax.annotate(r'$\alpha_t$', (x_t1 + 0.5 * np.cos(theta_t + alpha_t/2), y_t1 + 0.5 * np.sin(theta_t + alpha_t/2)),
             textcoords="offset points", xytext=(10,10), ha='center', color='red')

plt.grid(True)
plt.legend()
plt.show()