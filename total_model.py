import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the datasets
train_df = pd.read_csv('Data/total_three.csv')
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

# Normalize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validation_scaled = scaler.transform(X_validation)

# Reshape the input data for LSTM (assuming sequence length is 1)
X_train_lstm = X_train_scaled.reshape(
    (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape(
    (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
X_validation_lstm = X_validation_scaled.reshape(
    (X_validation_scaled.shape[0], 1, X_validation_scaled.shape[1]))

# Build and train the LSTM model using TensorFlow/Keras
model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='relu',
                         input_shape=(1, X_train_scaled.shape[1])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=2)

# Make predictions for the validation set
predictions_validation_lstm = model_lstm.predict(X_validation_lstm)

# Calculate Mean Squared Error on the validation set
mse_validation_lstm = mean_squared_error(
    y_validation, predictions_validation_lstm)
print(f'Mean Squared Error on Validation Set (LSTM): {mse_validation_lstm}')

# Save the trained LSTM model in HDF5 format
model_lstm.save('lstm_model_total')

# Get the layer before the last
# Assuming the layer before the last is at index -2
layer_before_last = model_lstm.layers[-2]

# Get the weights of the layer
weights_before_last = layer_before_last.get_weights()

# Check the shape of each element in weights_before_last
for i, weight in enumerate(weights_before_last):
    print(f'Shape of weight {i}: {weight.shape}')

# If they have different shapes, you might need to handle each element separately
# For example, you can save each element in a separate file
for i, weight in enumerate(weights_before_last):
    np.save(f'weights_before_last_{i}_total.npy', weight)

# Save mean and scale values
np.save('scaler_mean_total.npy', scaler.mean_)
np.save('scaler_scale_total.npy', scaler.scale_)
