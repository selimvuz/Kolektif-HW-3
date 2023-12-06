import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('Data/onlyA.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by 'Date'
df.sort_values(by='Date', inplace=True)

# Feature Engineering
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['quarter']  # Assuming 'quarter' is already a quarter feature

# Select features and target variable
features = df[['Open', 'Volume', 'DayOfWeek', 'Month', 'Quarter']]
target = df['Close']

# Normalize numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the dataset into training and testing sets
split_date = pd.to_datetime('2020-01-02')
train_data = df[df['Date'] < split_date]
test_data = df[df['Date'] >= split_date]

X_train, X_test = features_scaled[:len(
    train_data)], features_scaled[len(train_data):]
y_train, y_test = target[:len(train_data)], target[len(train_data):]

# Build and train the ANN model using TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu',
                          input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

# Make predictions for the testing set
predictions = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error on Test Set: {mse}')

# Save the trained model in HDF5 format
model.save('ann_model')

# Save mean and scale values
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)
