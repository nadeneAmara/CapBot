import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random
from data_prep import *

def create_model(maxLen, num_features):
    model = Sequential()
    model.add(Embedding(num_features, 50, input_length=maxLen))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_features, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

#text = load_text()
#s = get_sequences(text)
filename = 'reddit_comments.txt'
sequences = load_doc(filename)
sequences = sequences.split('\n')
x, y, maxLen, num_features = prepare_sequences(sequences)
model = create_model(maxLen, num_features)

path = F"/content/drive/MyDrive/captbot.ckpt" 
#Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=False,
                                                 verbose=1)
model.fit(x, y, epochs=150, callbacks=[cp_callback])

def generate_seq_diverse(model, tokenizer, seq_length, seed_text, n_words):
  result = list()
  in_text = seed_text
    # generate a fixed number of words
  for _ in range(n_words):
        # encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
    probabilities = model.predict(encoded)
    predictions = []
    for word, index in tokenizer.word_index.items():
      predictions.append({'text': in_text + ' ' + word, 
                          'score': probabilities[0][index]})
    predictions = sorted(predictions, key=lambda p: p['score'], reverse=True)
    top_predictions = []
    top_score = predictions[0]['score']
    min_score = 0.6
    rand_value = random.randint(int(min_score * 1000),1000)
    for p in predictions:
      if p['score'] >= rand_value/1000*top_score:
        top_predictions.append(p)
    random.shuffle(top_predictions)
    in_text = top_predictions[0]['text']
  return in_text

# load model 
new_model = tf.keras.models.load_model("/content/drive/My Drive/captbot.ckpt")
# load tokenizer
tokenizer = pickle.load(open('/content/drive/My Drive/tokenizer.pkl', 'rb'))

