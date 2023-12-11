import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import keras

# Load the array from the .npy file
loaded_array = np.load(
    'Weights/weights_before_last_0_common.npy', allow_pickle=True)
loaded_array_2 = np.load(
    'Weights/weights_before_last_1_common.npy', allow_pickle=True)

# Print the array
print("Size of arrays: ", loaded_array.shape, loaded_array_2.shape)

model_lstm = keras.models.load_model('lstm_model_m1')

print(model_lstm.layers[-2].get_weights())
