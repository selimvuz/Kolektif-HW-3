import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import pandas as pd
import pickle
from keras.regularizers import l2
import random

random.seed(42)
tf.random.set_seed(42)

# Replace with your actual file paths
train_path = 'Data/Train.csv'
valid_path = 'Data/Valid.csv'
test_path = 'Data/Test.csv'

train_data = pd.read_csv(train_path)
valid_data = pd.read_csv(valid_path)
test_data = pd.read_csv(test_path)

# Tokenization
tokenizer = Tokenizer(num_words=100000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['text'])

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_data['text'])
valid_sequences = tokenizer.texts_to_sequences(valid_data['text'])
test_sequences = tokenizer.texts_to_sequences(test_data['text'])

# Padding
max_length = 100  # You can adjust this
train_padded = pad_sequences(
    train_sequences, maxlen=max_length, padding='post', truncating='post')
valid_padded = pad_sequences(
    valid_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding='post', truncating='post')

train_labels = train_data['label'].values
valid_labels = valid_data['label'].values
test_labels = test_data['label'].values

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(100000, 16, input_length=max_length),
    tf.keras.layers.SimpleRNN(4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001)  # Set your learning rate here
model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=['accuracy'])

batch_size = 64  # Set your batch size here

early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(train_padded, train_labels, batch_size=batch_size, epochs=5, validation_data=(
    valid_padded, valid_labels), callbacks=[early_stop])

test_loss, test_acc = model.evaluate(test_padded, test_labels)
print('Test accuracy:', test_acc)

# Save the model
model.save('M3_model')

# Save the tokenizer
with open('final_tokenizerM3.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
