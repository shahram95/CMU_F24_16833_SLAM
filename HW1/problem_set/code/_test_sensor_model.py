import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from map_reader import MapReader
from sensor_model import SensorModel

class InteractiveSensorModel:
    def __init(self, map_path):
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

    def update(self):
        pass

    def reset(self):
        pass

    def simulate_laser_scan(self):
        pass

    def format_prob(self):
        pass


if __name__ == "__main__":
    map_path = '../data/map/wean.dat'
    interactive_model = InteractiveSensorModel(map_path)
    interactive_model.show()