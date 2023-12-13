import numpy as np
import matplotlib.pyplot as plt

# Updated alpha and loss values for the extended extrapolation range
alphas_updated = np.array(
    [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 1.5, 2.0, 2.5, 3.0])
losses_updated = np.array(
    [4.2752, 3.8003, 3.0518, 2.3981, 1.8041, 1.2576, 1.2953, 1.4491, 1.9936, 2.6900])

# Creating the updated line graph
plt.figure(figsize=(10, 6))
plt.plot(alphas_updated, losses_updated,
         marker='o', color='black', linestyle='-')

# Highlighting the interpolation range (0 to 1)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)
plt.fill_betweenx([min(losses_updated), max(losses_updated)],
                  0, 1, color='gray', alpha=0.2)

plt.title('Alpha ile Loss değişimi (Ekstrapolasyon)')
plt.xlabel('Alpha')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
