import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.metrics import mean_squared_error


def prepare_input(open_price, volume, day_of_week, month, quarter, scaler):
    input_data = pd.DataFrame({'Open': [open_price],
                               'Volume': [volume],
                               'DayOfWeek': [day_of_week],
                               'Month': [month],
                               'Quarter': [quarter]})

    # Scale the input features using the same scaler
    input_scaled = scaler.transform(input_data)

    # Reshape the input to match the expected shape (assuming LSTM input shape)
    input_reshaped = input_scaled.reshape(
        (input_scaled.shape[0], 1, input_scaled.shape[1]))

    return input_reshaped


def predict_closing_price(open_price, volume, day_of_week, month, quarter, model, scaler):
    # Prepare input data
    input_scaled = prepare_input(
        open_price, volume, day_of_week, month, quarter, scaler)

    # Make predictions using the trained model
    predicted_price = model.predict(input_scaled)[0][0]

    return predicted_price


if __name__ == "__main__":
    # Load the trained ANN model
    model = keras.models.load_model('lstm_model_m2')

    # Load the scaler used during training
    scaler = StandardScaler()  # Initialize a new scaler
    # Load mean from saved file
    scaler.mean_ = np.load('Scaler/scaler_mean_m2.npy')
    # Load scale from saved file
    scaler.scale_ = np.load('Scaler/scaler_scale_m2.npy')

    actual_closing_price = 153.0

    # Example input parameters (replace with your values)
    open_price = 150.0
    volume = 2000000
    day_of_week = 2
    month = 5
    quarter = 2

    # Convert numerical values to date format
    input_date = f"{day_of_week}.{month}.2023"

    # Predict closing price
    predicted_price = predict_closing_price(
        open_price, volume, day_of_week, month, quarter, model, scaler)

    # Calculate Mean Squared Error
    mse = mean_squared_error([actual_closing_price], [predicted_price])

    print(f"For the date {input_date}\nFor the opening price: {open_price}\nFor the volume: {volume}\nThe Predicted Closing Price is: {predicted_price}")
    print(f"Actual Closing Price: {actual_closing_price}")
    print(f"Mean Squared Error: {mse}")
