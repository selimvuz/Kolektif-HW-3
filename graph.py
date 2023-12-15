from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

# Data from the user
data = [
    {"alpha": -3.0, "loss": 0.6305, "accuracy": 0.7420},
    {"alpha": -2.5, "loss": 0.6343, "accuracy": 0.7406},
    {"alpha": -2.0, "loss": 0.6378, "accuracy": 0.7354},
    {"alpha": -1.5, "loss": 0.6383, "accuracy": 0.7382},
    {"alpha": -1.0, "loss": 0.6356, "accuracy": 0.7438},
    {"alpha": -0.5, "loss": 0.6320, "accuracy": 0.7480},
    {"alpha": 1.5, "loss": 0.6296, "accuracy": 0.7530},
    {"alpha": 2.0, "loss": 0.6315, "accuracy": 0.7500},
    {"alpha": 2.5, "loss": 0.6302, "accuracy": 0.7506},
    {"alpha": 3.0, "loss": 0.6266, "accuracy": 0.7558}
]

# Extracting alpha, loss, and accuracy
alphas = [d['alpha'] for d in data]
losses = [d['loss'] for d in data]
accuracies = [d['accuracy'] for d in data]


# Creating a grid on which to interpolate
grid_x, grid_y = np.meshgrid(np.linspace(min(alphas), max(
    alphas), 100), np.linspace(min(accuracies), max(accuracies), 100))

# Interpolating the loss values on the grid
grid_z = griddata((alphas, accuracies), losses,
                  (grid_x, grid_y), method='cubic')

# Creating the contour plot
plt.figure(figsize=(8, 6))
cp = plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap='viridis')
plt.colorbar(cp)  # Add a colorbar to a plot
plt.scatter(alphas, accuracies, color='red')  # Plot the actual data points
plt.title('Ekstrapolasyonda Doğruluk ve Kayıp Değerleri')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
