import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Extracting needed values from the provided text and adding them to the data list
import re

text = """
Alpha: 0.00, Beta: 0.00, Gamma: 1.00, Loss: 0.8057, Accuracy: 0.5696
Alpha: 0.00, Beta: 0.10, Gamma: 0.90, Loss: 0.6943, Accuracy: 0.5358
Alpha: 0.00, Beta: 0.20, Gamma: 0.80, Loss: 0.6914, Accuracy: 0.5208
Alpha: 0.00, Beta: 0.30, Gamma: 0.70, Loss: 0.6924, Accuracy: 0.5110
Alpha: 0.00, Beta: 0.40, Gamma: 0.60, Loss: 0.6929, Accuracy: 0.5030
Alpha: 0.00, Beta: 0.50, Gamma: 0.50, Loss: 0.6931, Accuracy: 0.5086
Alpha: 0.00, Beta: 0.60, Gamma: 0.40, Loss: 0.6930, Accuracy: 0.5146
Alpha: 0.00, Beta: 0.70, Gamma: 0.30, Loss: 0.6925, Accuracy: 0.5216
Alpha: 0.00, Beta: 0.80, Gamma: 0.20, Loss: 0.6917, Accuracy: 0.5250
Alpha: 0.00, Beta: 0.90, Gamma: 0.10, Loss: 0.7185, Accuracy: 0.5480
Alpha: 0.00, Beta: 1.00, Gamma: 0.00, Loss: 1.1091, Accuracy: 0.5620
Alpha: 0.10, Beta: 0.00, Gamma: 0.90, Loss: 0.6918, Accuracy: 0.5254
Alpha: 0.10, Beta: 0.10, Gamma: 0.80, Loss: 0.6921, Accuracy: 0.5130
Alpha: 0.10, Beta: 0.20, Gamma: 0.70, Loss: 0.6927, Accuracy: 0.5038
Alpha: 0.10, Beta: 0.30, Gamma: 0.60, Loss: 0.6930, Accuracy: 0.5016
Alpha: 0.10, Beta: 0.40, Gamma: 0.50, Loss: 0.6931, Accuracy: 0.5082
Alpha: 0.10, Beta: 0.50, Gamma: 0.40, Loss: 0.6930, Accuracy: 0.5140
Alpha: 0.10, Beta: 0.60, Gamma: 0.30, Loss: 0.6928, Accuracy: 0.5196
Alpha: 0.10, Beta: 0.70, Gamma: 0.20, Loss: 0.6921, Accuracy: 0.5224
Alpha: 0.10, Beta: 0.80, Gamma: 0.10, Loss: 0.6907, Accuracy: 0.5166
Alpha: 0.10, Beta: 0.90, Gamma: 0.00, Loss: 0.6997, Accuracy: 0.5382
Alpha: 0.20, Beta: 0.00, Gamma: 0.80, Loss: 0.6927, Accuracy: 0.5052
Alpha: 0.20, Beta: 0.10, Gamma: 0.70, Loss: 0.6929, Accuracy: 0.5046
Alpha: 0.20, Beta: 0.20, Gamma: 0.60, Loss: 0.6930, Accuracy: 0.5022
Alpha: 0.20, Beta: 0.30, Gamma: 0.50, Loss: 0.6931, Accuracy: 0.5054
Alpha: 0.20, Beta: 0.40, Gamma: 0.40, Loss: 0.6931, Accuracy: 0.5030
Alpha: 0.20, Beta: 0.50, Gamma: 0.30, Loss: 0.6930, Accuracy: 0.5126
Alpha: 0.20, Beta: 0.60, Gamma: 0.20, Loss: 0.6927, Accuracy: 0.5160
Alpha: 0.20, Beta: 0.70, Gamma: 0.10, Loss: 0.6918, Accuracy: 0.5172
Alpha: 0.20, Beta: 0.80, Gamma: 0.00, Loss: 0.6912, Accuracy: 0.5332
Alpha: 0.30, Beta: 0.00, Gamma: 0.70, Loss: 0.6931, Accuracy: 0.5060
Alpha: 0.30, Beta: 0.10, Gamma: 0.60, Loss: 0.6931, Accuracy: 0.5054
Alpha: 0.30, Beta: 0.20, Gamma: 0.50, Loss: 0.6931, Accuracy: 0.5000
Alpha: 0.30, Beta: 0.30, Gamma: 0.40, Loss: 0.6931, Accuracy: 0.5032
Alpha: 0.30, Beta: 0.40, Gamma: 0.30, Loss: 0.6931, Accuracy: 0.5006
Alpha: 0.30, Beta: 0.50, Gamma: 0.20, Loss: 0.6930, Accuracy: 0.5092
Alpha: 0.30, Beta: 0.60, Gamma: 0.10, Loss: 0.6926, Accuracy: 0.5096
Alpha: 0.40, Beta: 0.00, Gamma: 0.60, Loss: 0.6932, Accuracy: 0.5058
Alpha: 0.40, Beta: 0.10, Gamma: 0.50, Loss: 0.6931, Accuracy: 0.5002
Alpha: 0.40, Beta: 0.20, Gamma: 0.40, Loss: 0.6931, Accuracy: 0.5034
Alpha: 0.40, Beta: 0.30, Gamma: 0.30, Loss: 0.6931, Accuracy: 0.5048
Alpha: 0.40, Beta: 0.40, Gamma: 0.20, Loss: 0.6931, Accuracy: 0.5054
Alpha: 0.40, Beta: 0.50, Gamma: 0.10, Loss: 0.6930, Accuracy: 0.5050
Alpha: 0.50, Beta: 0.00, Gamma: 0.50, Loss: 0.6932, Accuracy: 0.4986
Alpha: 0.50, Beta: 0.10, Gamma: 0.40, Loss: 0.6931, Accuracy: 0.5036
Alpha: 0.50, Beta: 0.20, Gamma: 0.30, Loss: 0.6931, Accuracy: 0.5060
Alpha: 0.50, Beta: 0.30, Gamma: 0.20, Loss: 0.6931, Accuracy: 0.5036
Alpha: 0.50, Beta: 0.40, Gamma: 0.10, Loss: 0.6931, Accuracy: 0.5022
Alpha: 0.50, Beta: 0.50, Gamma: 0.00, Loss: 0.6931, Accuracy: 0.5162
Alpha: 0.60, Beta: 0.00, Gamma: 0.40, Loss: 0.6931, Accuracy: 0.5008
Alpha: 0.60, Beta: 0.10, Gamma: 0.30, Loss: 0.6931, Accuracy: 0.5054
Alpha: 0.60, Beta: 0.20, Gamma: 0.20, Loss: 0.6931, Accuracy: 0.5074
Alpha: 0.60, Beta: 0.30, Gamma: 0.10, Loss: 0.6932, Accuracy: 0.4974
Alpha: 0.70, Beta: 0.00, Gamma: 0.30, Loss: 0.6930, Accuracy: 0.5134
Alpha: 0.70, Beta: 0.10, Gamma: 0.20, Loss: 0.6929, Accuracy: 0.5110
Alpha: 0.70, Beta: 0.20, Gamma: 0.10, Loss: 0.6930, Accuracy: 0.5070
Alpha: 0.80, Beta: 0.00, Gamma: 0.20, Loss: 0.6927, Accuracy: 0.5090
Alpha: 0.80, Beta: 0.10, Gamma: 0.10, Loss: 0.6924, Accuracy: 0.5246
Alpha: 0.90, Beta: 0.00, Gamma: 0.10, Loss: 0.6917, Accuracy: 0.5242
Alpha: 1.00, Beta: 0.00, Gamma: 0.00, Loss: 0.9676, Accuracy: 0.5336
"""

# Regular expression pattern to extract the needed values
pattern = r"Alpha: ([\d.]+), Beta: ([\d.]+), Gamma: ([\d.]+), Loss: ([\d.]+), Accuracy: ([\d.]+)"

# Extracting the values
extracted_data = re.findall(pattern, text)

# Converting strings to floats and adding them to the data list
data = [(float(alpha), float(beta), float(gamma), float(loss), float(acc))
        for alpha, beta, gamma, loss, acc in extracted_data]

# Unpacking data
alphas, betas, gammas, losses, accs = zip(*data)

# Creating the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting
sc = ax.scatter(alphas, betas, accs, c=losses, cmap='viridis', marker='o')

# Labels and title
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('Doğruluk')
ax.set_title('3D Ağırlık Uzayında Doğruluk')

# Color bar
plt.colorbar(sc)

plt.show()
