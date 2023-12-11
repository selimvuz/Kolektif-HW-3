import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import keras

# Load the weights
weights1 = np.load('Weights/weights_before_last_0_m1.npy')
weights2 = np.load('Weights/weights_before_last_0_m2.npy')
biases1 = np.load('Weights/weights_before_last_1_m1.npy')
biases2 = np.load('Weights/weights_before_last_1_m2.npy')

# Load the datasets
train_df = pd.read_csv('Data/common_data.csv')
test_df = pd.read_csv('Data/test_data.csv')
validation_df = pd.read_csv('Data/validation.csv')

# Feature Engineering

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def feature_engineering(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    # Assuming 'quarter' is already a quarter feature
    df['Quarter'] = df['quarter']
    return df


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)
validation_df = feature_engineering(validation_df)

# Select features and target variable
features = ['Open', 'Volume', 'DayOfWeek', 'Month', 'Quarter']
target = 'Close'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]
X_validation, y_validation = validation_df[features], validation_df[target]

N = 30  # Number of points
interpolated_weights_list = []
interpolated_biases_list = []

for i in range(N):
    lambda_param = i / (N - 1)  # Adjust based on the desired range

    # Perform linear interpolation for weights
    interpolated_weights = lambda_param * \
        weights1 + (1 - lambda_param) * weights2
    interpolated_weights_list.append(interpolated_weights)

    # Perform linear interpolation for biases
    interpolated_bias = lambda_param * biases1 + (1 - lambda_param) * biases2
    interpolated_biases_list.append(interpolated_bias)

scaler_m1 = StandardScaler()
scaler_m1.mean_ = np.load('Scaler/scaler_mean_m1.npy')
scaler_m1.scale_ = np.load('Scaler/scaler_scale_m1.npy')

scaler_m2 = StandardScaler()
scaler_m2.mean_ = np.load('Scaler/scaler_mean_m2.npy')
scaler_m2.scale_ = np.load('Scaler/scaler_scale_m2.npy')

# Scale the validation data using the loaded scaler
X_val_scaled_m1 = scaler_m1.transform(X_validation)
X_val_scaled_m2 = scaler_m2.transform(X_validation)

# Create the LSTM model
model_lstm = keras.models.load_model('lstm_model_m1')

model_lstm.compile(optimizer='adam', loss='mean_squared_error')

pre_model_m1 = keras.models.load_model('lstm_model_m1')
pre_model_m2 = keras.models.load_model('lstm_model_m2')

# Reshape validation data for LSTM
X_val_reshaped = X_val_scaled_m1.reshape(
    (X_val_scaled_m1.shape[0], 1, X_val_scaled_m1.shape[1]))
# Reshape validation data for LSTM
X_val_reshaped_m2 = X_val_scaled_m2.reshape(
    (X_val_scaled_m2.shape[0], 1, X_val_scaled_m2.shape[1]))

pre_mse_m1 = pre_model_m1.evaluate(X_val_reshaped, y_validation, verbose=0)
pre_mse_m2 = pre_model_m2.evaluate(X_val_reshaped_m2, y_validation, verbose=0)

print(f'Pre-interpolation MSE: {pre_mse_m1}')
print(f'Pre-interpolation MSE: {pre_mse_m2}')

# Evaluate Performance at Each Interpolated Point
performance_metrics = []

for i, (interpolated_weights, interpolated_biases) in enumerate(zip(interpolated_weights_list, interpolated_biases_list)):
    # Set the interpolated weights and biases to the model
    model_lstm.layers[-2].set_weights([interpolated_weights,
                                      interpolated_biases])

    # Compile the model
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Evaluate model performance
    mse = model_lstm.evaluate(X_val_reshaped, y_validation, verbose=0)
    mse_2 = model_lstm.evaluate(X_val_reshaped_m2, y_validation, verbose=0)
    performance_metrics.append(mse + mse_2 / 2)

    # Print the MSE for each point
    print(f'Point {i + 1}: MSE = {mse}')
