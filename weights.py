import tensorflow as tf
import numpy as np
import os

model_path = 'M2_model'
model = tf.keras.models.load_model(model_path)

save_dir = 'M2_model_weights'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, layer in enumerate(model.layers):
    layer_weights = layer.get_weights()
    if layer_weights:  # Check if the layer has weights
        layer_dir = os.path.join(save_dir, f'layer_{i}_{layer.name}')
        os.makedirs(layer_dir, exist_ok=True)
        for j, weight in enumerate(layer_weights):
            weight_file = os.path.join(layer_dir, f'weight_{j}.npy')
            np.save(weight_file, weight)
