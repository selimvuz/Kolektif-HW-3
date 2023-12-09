import numpy as np

# Load the array from the .npy file
loaded_array = np.load('weights_before_last_0_common.npy', allow_pickle=True)
loaded_array_2 = np.load('weights_before_last_1_common.npy', allow_pickle=True)

# Print the array
print("Size of arrays: ", loaded_array.shape, loaded_array_2.shape)
