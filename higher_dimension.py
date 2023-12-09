import tensorflow as tf
from keras import layers

# Define a simple autoencoder-like architecture


def create_autoencoder(input_dim, encoding_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = tf.keras.Model(input_layer, decoded)

    return autoencoder


# Assuming your model's final weights have 1000 dimensions
input_dim = 1000
encoding_dim = 10  # Choose a lower-dimensional space for the bottleneck

# Create two autoencoder models
model1 = create_autoencoder(input_dim, encoding_dim)
model2 = create_autoencoder(input_dim, encoding_dim)

# Interpolation parameter (lambda)
lambda_param = 0.5

# Interpolate between the final weights
interpolated_weights = [
    lambda_param * w1 + (1 - lambda_param) * w2
    for w1, w2 in zip(model1.get_weights(), model2.get_weights())
]

# Set the interpolated weights to a new model
interpolated_model = create_autoencoder(input_dim, encoding_dim)
interpolated_model.set_weights(interpolated_weights)

# After training, get the final weights
final_weights = interpolated_model.get_weights()

# Print the shapes of the final weights
for i, weight in enumerate(final_weights):
    print(f"Layer {i + 1} - Shape: {weight.shape}")
