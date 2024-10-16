import numpy as np
from ekf_slam import warp2pi

angles = np.array([-4*np.pi, -np.pi, -3*np.pi/2, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 4*np.pi])

for angle in angles:
    warped_angles = warp2pi(angle)

    print("Original Angles:", angle)
    print("Warped Angles:   ", warped_angles)
