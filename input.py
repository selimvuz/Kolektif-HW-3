import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def prepare_input(open_price, volume, day_of_week, month, quarter, scaler):
    input_data = pd.DataFrame({'Open': [open_price],
                               'Volume': [volume],
                               'DayOfWeek': [day_of_week],
                               'Month': [month],
                               'Quarter': [quarter]})

    # Scale the input features using the same scaler
    input_scaled = scaler.transform(input_data)
    return input_scaled


def predict_closing_price(open_price, volume, day_of_week, month, quarter, model, scaler):
    # Prepare input data
    input_scaled = prepare_input(
        open_price, volume, day_of_week, month, quarter, scaler)

    # Make predictions using the trained model
    predicted_price = model.predict(input_scaled)[0][0]

    return predicted_price


if __name__ == "__main__":
    # Load the trained ANN model
    model = keras.models.load_model('ann_model')

    # Load the scaler used during training
    scaler = StandardScaler()  # Initialize a new scaler
    scaler.mean_ = np.load('scaler_mean.npy')  # Load mean from saved file
    scaler.scale_ = np.load('scaler_scale.npy')  # Load scale from saved file

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

    print(f"For the date {input_date}\nFor the opening price: {open_price}\nFor the volume: {volume}\nThe Predicted Closing Price is: {predicted_price}")
