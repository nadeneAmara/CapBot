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
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc
from clarifai_grpc.grpc.api import service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())

# Build a model with Embedding Layer and single LSTM layer
def create_model(maxLen, num_features):
    model = Sequential()
    model.add(Embedding(num_features, 50, input_length=maxLen))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_features, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Build dataset and split into sequences for model
def train_model():
    # If 'reddit_comments.txt' does not exist, generate corpus
    text = load_text()
    s = get_sequences(text)
    filename = 'reddit_comments.txt'
    sequences = load_doc(filename)
    sequences = sequences.split('\n')
    x, y, maxLen, num_features = prepare_sequences(sequences)
    model = create_model(maxLen, num_features)

    path = F"model/captbot.ckpt" 
    #Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=False,
                                                 verbose=1)
    model.fit(x, y, epochs=150, callbacks=[cp_callback])

# First method of generating sequences, append prediction for each word to result
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # Generate a fixed number of words
    for _ in range(n_words):
        # Encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # Truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # Predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # Map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # Append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

# Second method, consider all predictions and take one of better predictions at random
def generate_seq_diverse(model, tokenizer, seq_length, seed_text, n_words):
  result = list()
  in_text = seed_text
    # Generate a fixed number of words
  for _ in range(n_words):
        # Encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
        # Truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # Predict probabilities for each word
    probabilities = model.predict(encoded)
    predictions = []
    # Generate all possible predictions
    for word, index in tokenizer.word_index.items():
      predictions.append({'text': in_text + ' ' + word, 
                          'score': probabilities[0][index]})
    predictions = sorted(predictions, key=lambda p: p['score'], reverse=True)
    top_predictions = []
    top_score = predictions[0]['score']
    # Predictions must have minimal score of 0.35
    min_score = 0.35
    rand_value = random.randint(int(min_score * 1000),1000)
    # If predictions meets minimal score, add to potential final predictions
    for p in predictions:
      if p['score'] >= rand_value/1000*top_score:
        top_predictions.append(p)
    # Take one at random
    random.shuffle(top_predictions)
    in_text = top_predictions[0]['text']
  return in_text

# Filter Clarifai keywords by type using NLTK, construct seed text using adj + noun + verb structure
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
    # Pick words at random to have variety
    i = random.randint(0,(min_length-1))
    seed_text = 'The' + ' ' + adj[i] + ' ' + nouns[i] + ' ' + verbs[i]
    return seed_text

# Input an image URL to grab keywords using Clarifai. Then generate their tags using NlTK.
def prepare_keywords(image):
    #app = ClarifaiApp(api_key='a5288575bd8b453285d62995dd09cb9a')
    #model_clarifai = app.public_models.general_model
    metadata = (('authorization', 'Key a5288575bd8b453285d62995dd09cb9a'),)
    request = service_pb2.PostModelOutputsRequest(
    # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
    model_id='aaa03c23b3724a16a56b629203edc62c',
    inputs=[
      resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=image)))
    ])
    response = stub.PostModelOutputs(request, metadata=metadata)

    response = model_clarifai.predict_by_url(image)
    keywords = []
    #for dict_item in response['outputs'][0]['data']['concepts']:
    for dict_item in response.outputs[0].data.concepts:
        #keywords.append(dict_item['name'])
        keywords.append(dict_item.name)
    str1 = " ".join(keywords)
    text1 = nltk.word_tokenize(str1)
    tags = nltk.pos_tag(text1)
    return tags

def main(url):
    image = url
    tags = prepare_keywords(image)
    seed = chooseSeedText(tags)
    # load model 
    new_model = tf.keras.models.load_model("model/captbot.ckpt")
    # load tokenizer
    tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))
    generated2 = generate_seq_diverse(new_model, tokenizer, 114, seed, 7)
    generated = generate_seq(new_model, tokenizer, 114, seed, 7)
    return generated2

