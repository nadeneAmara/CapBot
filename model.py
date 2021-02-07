import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random
from data_prep import *
from clarifai import rest
from clarifai.rest import ClarifaiApp
import nltk

def create_model(maxLen, num_features):
    model = Sequential()
    model.add(Embedding(num_features, 50, input_length=maxLen))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_features, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
'''
text = load_text()
s = get_sequences(text)
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
'''
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

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
    print(probabilities)
    predictions = []
    for word, index in tokenizer.word_index.items():
      predictions.append({'text': in_text + ' ' + word, 
                          'score': probabilities[0][index]})
    predictions = sorted(predictions, key=lambda p: p['score'], reverse=True)
    #print(predictions)
    top_predictions = []
    top_score = predictions[0]['score']
    print(top_score)
    print("top_score: %d " % top_score)
    min_score = 0.35
    rand_value = random.randint(int(min_score * 1000),1000)
    print("rand_value: %d " % rand_value)
    for p in predictions:
      if p['score'] >= rand_value/1000*top_score:
        print(rand_value/1000*top_score)
        print("greater than or equal to: %d " % (rand_value/1000*top_score))
        print("p_score: %d " % p['score'])
        top_predictions.append(p)
    random.shuffle(top_predictions)
    in_text = top_predictions[0]['text']
  return in_text

def chooseSeedText(keywords):
    adj = []
    nouns = []
    verbs = []
    for word, tag in keywords:
        if (tag == 'JJ'):
            adj.append(word)
        elif (tag == 'NN'):
            nouns.append(word)
        elif (tag == 'VB' or tag == 'VBP' or tag == 'VBG'):
            verbs.append(word)
    min_length = min(len(adj),len(nouns),len(verbs))
    i = random.randint(0,(min_length-1))
    seed_text = adj[i] + ' ' + nouns[i] + ' ' + verbs[i]
    return seed_text

image = "https://www.ctvnews.ca/polopoly_fs/1.5098407.1599687805!/httpImage/image.jpg_gen/derivatives/landscape_1020/image.jpg"
app = ClarifaiApp(api_key='6d610d6e33da4541b836c1cd0fff34f7')
model_clarifai = app.public_models.general_model
response = model_clarifai.predict_by_url(image)
keywords = []
for dict_item in response['outputs'][0]['data']['concepts']:
    keywords.append(dict_item['name'])
str1 = " ".join(keywords)
text1 = nltk.word_tokenize(str1)
tags = nltk.pos_tag(text1)
print(tags)
seed = chooseSeedText(tags)
print(seed)

for i in keywords:
    print(i)
# load model 
new_model = tf.keras.models.load_model("model/captbot.ckpt")
# load tokenizer
tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))
#generated2 = generate_seq_diverse(new_model, tokenizer, 114, "The blue girl", 20)
#generated = generate_seq(new_model, tokenizer, 114, "The blue girl", 20)
#print(generated2)

