import numpy as np
import tensorflow as tf
from tensorflow import keras

def create_model(x, y, maxLen, num_features):
    model = sequential()
    model.add(LSTM(128, input_shape=(maxLen, num_features)))
    model.add(Dense(num_features, activation='softmax'))