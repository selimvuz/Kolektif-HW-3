import tensorflow as tf
import pandas as pd
import os
import pickle
from keras.preprocessing.sequence import pad_sequences

# Load the original models
model_M1 = tf.keras.models.load_model('M1_model')
model_M2 = tf.keras.models.load_model('M2_model')

max_length = 100  # Adjust as needed

# Load both tokenizers
with open('final_tokenizerM1.pickle', 'rb') as handle:
    tokenizer_negative = pickle.load(handle)

with open('final_tokenizerM2.pickle', 'rb') as handle:
    tokenizer_positive = pickle.load(handle)

test_path = 'Data/Test.csv'
test_data = pd.read_csv(test_path)
test_labels = test_data['label'].values

# Directory to save extrapolated models
save_dir = 'extrapolated_models'
os.makedirs(save_dir, exist_ok=True)

# Extrapolation function


def extrapolate_weights(weights1, weights2, alpha):
    return [(1 - alpha) * w1 + alpha * w2 for w1, w2 in zip(weights1, weights2)]


# Extrapolation factors
extrapolation_factors = [-3.0, -2.5, -2.0, -1.5, -1, -0.5, 1.5, 2.0, 2.5, 3.0]

performance = []

for alpha in extrapolation_factors:
    # Choose the tokenizer based on alpha
    tokenizer = tokenizer_negative if alpha < 0 else tokenizer_positive

    # Prepare the test data with the chosen tokenizer
    test_sequences = tokenizer.texts_to_sequences(test_data['text'])
    test_padded = pad_sequences(
        test_sequences, maxlen=max_length, padding='post', truncating='post')

    # Extrapolate weights for each layer
    extrapolated_weights = [extrapolate_weights(model_M1.layers[i].get_weights(), model_M2.layers[i].get_weights(
    ), alpha) for i in range(len(model_M1.layers)) if model_M1.layers[i].get_weights()]

    # Create new model and set extrapolated weights
    extrapolated_model = tf.keras.models.clone_model(model_M1)
    extrapolated_model.set_weights(
        [w for layer_weights in extrapolated_weights for w in layer_weights])
    extrapolated_model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Evaluate the model
    loss, acc = extrapolated_model.evaluate(
        test_padded, test_labels, verbose=0)
    performance.append((alpha, loss, acc))

# Print performance of each model
for alpha, loss, acc in performance:
    print(f"Alpha: {alpha:.3f}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
