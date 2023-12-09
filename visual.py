import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Sample data in 4D (x, y, z, w)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 1, 7, 0])
z = np.array([5, 8, 3, 10, 2])
w = np.array([10, 5, 8, 12, 6])

# Create a regular grid for interpolation in 4D
xi, yi, zi, wi = np.meshgrid(
    np.linspace(min(x), max(x), 20),
    np.linspace(min(y), max(y), 20),
    np.linspace(min(z), max(z), 20),
    np.linspace(min(w), max(w), 20),
)

# Combine the coordinates into a single array
points = np.vstack([x, y, z, w]).T

# Interpolate the values on the regular grid
values = griddata(points, w, (xi, yi, zi, wi), method='linear')

# Plot the original data points and the interpolated values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=w, cmap='viridis', s=100, label='Original Data')
ax.scatter(xi, yi, zi, c=values, cmap='viridis',
           marker='o', s=10, label='Interpolated Values')
ax.legend()
plt.show()
