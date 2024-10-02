import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from map_reader import MapReader
from sensor_model import SensorModel

class InteractiveSensorModel:
    def __init__(self, map_path):
        self.map_reader = MapReader(map_path)
        self.occupancy_map = self.map_reader.get_map()
        self.sensor_model = SensorModel(self.occupancy_map)

        self.fig, self.ax = plt.subplots(figsize=(10,8))
        plt.subplots_adjust(bottom=0.25)

        self.map_img = self.ax.imshow(self.occupancy_map, cmap='Greys', origin='lower')
        self.robot_plot, = self.ax.plot([], [], 'ro', markersize=10)
        self.laser_lines = []

        self.ax.set_title("Interactive Sensor Model")

        # Find initial random free space for spawning robot
        free_space = np.where(self.occupancy_map == 0)
        init_idx = np.random.randint(0, len(free_space[0]))
        self.x = free_space[1][init_idx] * 10
        self.y = free_space[0][init_idx] * 10
        self.theta = 0

        # Create x, y, theta control sliders
        ax_x = plt.axes([0.2, 0.15, 0.6, 0.03])
        ax_y = plt.axes([0.2, 0.1, 0.6, 0.03])
        ax_theta = plt.axes([0.2, 0.05, 0.6, 0.03])

        self.slider_x = Slider(ax_x, 'X', 0, self.occupancy_map.shape[1]*10, valinit=self.x)
        self.slider_y = Slider(ax_y, 'Y', 0, self.occupancy_map.shape[0]*10, valinit=self.y)
        self.slider_theta = Slider(ax_theta, 'Theta', -np.pi, np.pi, valinit=self.theta)

        self.slider_x.on_changed(self.update)
        self.slider_y.on_changed(self.update)
        self.slider_theta.on_changed(self.update)

        # Create a reset button
        reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset)

        self.update(None)

    def update(self, val):
        x = self.slider_x.val
        y = self.slider_y.val
        theta = self.slider_theta.val
        
        # Check if the new position is in free space
        x_cell, y_cell = int(x / 10), int(y / 10)
        if 0 <= x_cell < self.occupancy_map.shape[1] and 0 <= y_cell < self.occupancy_map.shape[0]:
            if self.occupancy_map[y_cell, x_cell] == 0:
                self.x, self.y, self.theta = x, y, theta
        
        self.robot_plot.set_data([self.x/10], [self.y/10])

        # Simulate laser scan
        z_t1_arr = self.simulate_laser_scan()

        # Compute probability using sensor model
        prob = self.sensor_model.beam_range_finder_model(z_t1_arr, [self.x, self.y, self.theta])

        # Update title with probability
        self.ax.set_title(f"Interactive Sensor Model - Probability: {self.format_prob(prob)}")

        # Update laser visualization
        for line in self.laser_lines:
            line.remove()
        self.laser_lines.clear()

        for i in range(0, 180, 1):
            angle = self.theta + np.radians(i - 90)
            end_x = self.x + z_t1_arr[i] * np.cos(angle)
            end_y = self.y + z_t1_arr[i] * np.sin(angle)
            line, = self.ax.plot([self.x/10, end_x/10], [self.y/10, end_y/10], 'g-', alpha=0.1)
            self.laser_lines.append(line)
        
        self.fig.canvas.draw_idle()

    def reset(self, event):
        free_space = np.where(self.occupancy_map == 0)
        init_idx = np.random.randint(0, len(free_space[0]))
        self.x = free_space[1][init_idx] * 10
        self.y = free_space[0][init_idx] * 10
        self.theta = 0

        self.slider_x.reset()
        self.slider_y.reset()
        self.slider_theta.reset()

        self.update(None)

    def simulate_laser_scan(self, max_range=1000, num_beams=180):
        z_t1_arr = np.full(num_beams, max_range)
        for i in range(num_beams):
            angle = self.theta + np.radians(i - 90)
            for r in range(0, max_range, 10):
                x_end = self.x + r * np.cos(angle)
                y_end = self.y + r * np.sin(angle)
                x_cell, y_cell = int(x_end / 10), int(y_end / 10)
                if (x_cell < 0 or x_cell >= self.occupancy_map.shape[1] or
                    y_cell < 0 or y_cell >= self.occupancy_map.shape[0] or
                    self.occupancy_map[y_cell, x_cell] > 0.5):
                    z_t1_arr[i] = r
                    break
        return z_t1_arr

    def format_prob(self, prob):
        if isinstance(prob, np.ndarray):
            return ', '.join(f'{p:.2e}' for p in prob.flat)
        elif isinstance(prob, (list, tuple)):
            return ', '.join(f'{p:.2e}' for p in prob)
        else:
            return f'{prob:.2e}'
    
    def show(self):
        plt.show()

if __name__ == "__main__":
    map_path = '../data/map/wean.dat'
    interactive_model = InteractiveSensorModel(map_path)
    interactive_model.show()