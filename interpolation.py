import tensorflow as tf
import numpy as np
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the original models
model_M1 = tf.keras.models.load_model('M1_model')
model_M2 = tf.keras.models.load_model('M2_model')

max_length = 100  # You can adjust this

train_path = 'Data/Train.csv'
test_path = 'Data/Test.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Tokenization
tokenizer = Tokenizer(num_words=100000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['text'])

test_sequences = tokenizer.texts_to_sequences(test_data['text'])
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding='post', truncating='post')
test_labels = test_data['label'].values

# Directory to save interpolated models
save_dir = 'extrapolated_models'
os.makedirs(save_dir, exist_ok=True)

# Function to interpolate between two sets of weights


def interpolate_weights(weights1, weights2, alpha):
    return [(1 - alpha) * w1 + alpha * w2 for w1, w2 in zip(weights1, weights2)]


# Create and evaluate interpolated models
num_interpolations = 200
performance = []

for i in range(num_interpolations + 1):
    alpha = i / num_interpolations

    # Interpolate weights for each layer
    interpolated_weights = []
    for layer_index in range(len(model_M1.layers)):
        weights_M1 = model_M1.layers[layer_index].get_weights()
        weights_M2 = model_M2.layers[layer_index].get_weights()

        if weights_M1 and weights_M2:  # If the layer has weights
            interpolated_weights.append(
                interpolate_weights(weights_M1, weights_M2, alpha))

    # Create a new model with the same architecture and set interpolated weights
    interpolated_model = tf.keras.models.clone_model(model_M1)
    interpolated_model.set_weights(
        [w for layer_weights in interpolated_weights for w in layer_weights])
    interpolated_model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Save each interpolated model
    model_save_path = os.path.join(
        save_dir, f'interpolated_model_alpha_{alpha:.2f}')
    interpolated_model.save(model_save_path)

    # Evaluate the model (assuming test data is available as test_padded and test_labels)
    loss, acc = interpolated_model.evaluate(
        test_padded, test_labels, verbose=0)
    performance.append((alpha, loss, acc))

# Print performance of each model
for alpha, loss, acc in performance:
    print(f"Alpha: {alpha:.3f}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
