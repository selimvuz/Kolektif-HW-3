import tensorflow as tf
import numpy as np
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

# Load the original models
model_M0 = tf.keras.models.load_model('M0_model')
model_M1 = tf.keras.models.load_model('M1_model')
model_M2 = tf.keras.models.load_model('M2_model')

max_length = 100  # You can adjust this

train_path = 'Data/Train.csv'
test_path = 'Data/Test.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Load both tokenizers
with open('final_tokenizerM1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

test_sequences = tokenizer.texts_to_sequences(test_data['text'])
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding='post', truncating='post')
test_labels = test_data['label'].values

# Directory to save interpolated models
save_dir = 'triangle_models'
os.makedirs(save_dir, exist_ok=True)

performance = []


def interpolate_barycentric(weights_M0, weights_M1, weights_M2, alpha, beta, gamma):
    return [alpha * w0 + beta * w1 + gamma * w2 for w0, w1, w2 in zip(weights_M0, weights_M1, weights_M2)]


def generate_barycentric_coordinates(step=0.1):
    coordinates = []
    for alpha in np.arange(0, 1 + step, step):
        for beta in np.arange(0, 1 - alpha + step, step):
            gamma = 1 - alpha - beta
            if gamma >= 0:
                coordinates.append((alpha, beta, gamma))
    return coordinates


barycentric_coords = generate_barycentric_coordinates(step=0.1)

for i, (alpha, beta, gamma) in enumerate(barycentric_coords):
    barycentric_weights = []
    for layer_index in range(len(model_M0.layers)):
        w_M0 = model_M0.layers[layer_index].get_weights()
        w_M1 = model_M1.layers[layer_index].get_weights()
        w_M2 = model_M2.layers[layer_index].get_weights()
        if w_M0 and w_M1 and w_M2:
            barycentric_weights.append(interpolate_barycentric(
                w_M0, w_M1, w_M2, alpha, beta, gamma))

    new_model = tf.keras.models.clone_model(model_M0)
    new_model.set_weights(
        [w for layer_weights in barycentric_weights for w in layer_weights])
    new_model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])

    # Save each model
    model_save_path = os.path.join(
        save_dir, f'model_{i}_alpha_{alpha:.2f}_beta_{beta:.2f}_gamma_{gamma:.2f}')
    new_model.save(model_save_path)

    # Evaluate the model
    loss, acc = new_model.evaluate(test_padded, test_labels, verbose=0)
    performance.append((alpha, beta, gamma, loss, acc))

# Print performance of each model
for alpha, beta, gamma, loss, acc in performance:
    print(
        f"Alpha: {alpha:.2f}, Beta: {beta:.2f}, Gamma: {gamma:.2f}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
