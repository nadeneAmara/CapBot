import numpy as np
import tensorflow as tf
from tensorflow import keras
from data_prep import *

def create_model(maxLen, num_features):
    model = Sequential()
    model.add(Embedding(num_features, 50, input_length=maxLen))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_features, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Load in The Prophet from Project Gutenberg 
url = "http://www.gutenberg.org/files/58585/58585-0.txt"
text = load_text(url)
sequences = get_sequences(text)
x, y, maxLen, num_features = prepare_sequences(sequences)
model = create_model(maxLen, num_features)
model.fit(x, y, epochs=100)