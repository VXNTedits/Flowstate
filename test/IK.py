import math

import glm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


pitch = np.linspace(glm.radians(-89), glm.radians(89), 1000)
yaw = np.linspace(glm.radians(-90), glm.radians(90), 1000)
pitch_grid, yaw_grid = np.meshgrid(pitch, yaw)

x0 = np.cos((pitch)) * np.cos((yaw))
y0 = np.sin((pitch))
z0 = np.cos((pitch)) * np.sin((yaw))

eye_to_shoulder = 0.314
arm_length = 0.812
shoulder_offset = 0.25

dx = shoulder_offset * np.cos((yaw))
dy = -eye_to_shoulder
dz = shoulder_offset * np.sin((yaw))
l = arm_length

A = dy - dx * y0
B = dy - dz * y0
E = l * y0 ** 2
C = (x0 * l) ** 2
Ap = 2 * x0 * l * A
D = (z0 * l) ** 2
Bp = 2 * z0 * l * B
alpha = -E ** 2 - C - D
beta = Ap - Bp
gamma = E ** 2 - A ** 2 - B ** 2

pitch_angle = np.asin((-beta + np.sqrt(beta ** 2 - 4 * alpha * gamma + 0j)) / (2 * alpha))
yaw_angle = np.asin(
    y0 * ((l * np.sin(pitch_angle) + dz) / (l * z0 * np.cos(pitch_angle))) - (dy / (l * np.cos(pitch_angle))))

# pitch_angle = np.asin((-((2 * np.cos((pitch)) * np.cos((yaw)) * l * (dy - dx * np.sin((pitch)))) - (2 * np.cos((pitch)) * np.sin((yaw)) * l * (dy - dz * np.sin((pitch)))))
#                        + np.sqrt(((2 * np.cos((pitch)) * np.cos((yaw)) * l * (dy - dx * np.sin((pitch)))) - (2 * np.cos((pitch)) * np.sin((yaw)) * l * (dy - dz * np.sin((pitch))))) ** 2
#                        - 4 * (-(l * np.sin((pitch)) ** 2) ** 2 - ((np.cos((pitch)) * np.cos((yaw)) * l) ** 2) - ((np.cos((pitch)) * np.sin((yaw)) * l) ** 2)) * ((l * np.sin((pitch)) ** 2) ** 2
#                        - (dy - dx * np.sin((pitch))) ** 2 - (dy - dz * np.sin((pitch))) ** 2) + 0j)) / (2 * (-(l * np.sin((pitch)) ** 2) ** 2
#                        - ((np.cos((pitch)) * np.cos((yaw)) * l) ** 2) - ((np.cos((pitch)) * np.sin((yaw)) * l) ** 2))))
#
# yaw_angle = np.asin(
#     np.sin((pitch)) * ((l * np.sin(pitch_angle) + dz) / (l * np.cos((pitch)) * np.sin((yaw)) * np.cos(pitch_angle))) - (dy / (l * np.cos(pitch_angle))))

pitch_angle_real = np.real(pitch_angle)
yaw_angle_real = np.real(yaw_angle)


print(f"pitch_grid shape: {pitch_grid.shape}")
print(f"yaw_grid shape: {yaw_grid.shape}")
print(f"pitch_angle_real shape: {pitch_angle_real.shape}")
print(f"yaw_angle_real shape: {yaw_angle_real.shape}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
contour_pitch = plt.contourf(pitch_grid, yaw_grid, pitch_angle_real, levels=100, cmap='viridis')
plt.colorbar(contour_pitch, label='Pitch Angle (degrees)')
plt.xlabel('Pitch Input (degrees)')
plt.ylabel('Yaw Input (degrees)')
plt.title('Computed Pitch Angle')

plt.subplot(1, 2, 2)
contour_yaw = plt.contourf(pitch_grid, yaw_grid, yaw_angle_real, levels=100, cmap='viridis')
plt.colorbar(contour_yaw, label='Yaw Angle (degrees)')
plt.xlabel('Pitch Input (degrees)')
plt.ylabel('Yaw Input (degrees)')
plt.title('Computed Yaw Angle')

plt.tight_layout()
plt.show()