import pandas as pd

# Load your dataset
df = pd.read_csv('Data/Train.csv')  # Replace with your file path

# Assuming 'text' is the column containing the text data
word_lengths = df['text'].str.split().apply(len)

# Calculate minimum and maximum word count
min_word_count = word_lengths.min()
max_word_count = word_lengths.max()

print("Minimum word count:", min_word_count)
print("Maximum word count:", max_word_count)
