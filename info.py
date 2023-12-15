import tensorflow as tf

model = tf.keras.models.load_model('M1_model')

# Initialize the count
total_weights = 0

# Iterate through each layer
for layer in model.layers:
    layer_weights = layer.get_weights()  # List of numpy arrays
    # Sum sizes of all weight arrays in the layer
    num_weights = sum([w.size for w in layer_weights])
    total_weights += num_weights

print(f"Total number of weights in the model: {total_weights}")
