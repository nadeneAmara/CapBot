from __future__ import division
import string
import nltk, re, pprint
from nltk import tokenize
from nltk import word_tokenize
from urllib import request
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from pickle import dump
import requests

# Retrieve comments from pushshift reddit database
def request_comments(**kwargs):
  response = requests.get("https://api.pushshift.io/reddit/comment/search/",params=kwargs)
  data = response.json()
  comments = data['data']
  return comments

# Get text from comments from given subreddits
def get_comment_set(comment_number):
  sr_names = ["aww","wholesomememes","funny", "interestingasfuck", "EarthPorn"]
  comment_bodies = ""
  before = None
  for i in sr_names:
    comments_left = comment_number
    while (comments_left > 0):
      comments = request_comments(subreddit=i, size=100, before=before, sort='desc',sort_type='created_utc')
      for comment in comments:
        comment_bodies = comment_bodies + comment['body']
        before = comment['created_utc']
      comments_left = comments_left - 100
      time.sleep(2)
  return comment_bodies

# Save comments to file, line by line
def save_doc(sequences, filename):
    dataset = '\n'.join(sequences)
    file = open(filename, 'w')
    file.write(dataset)
    file.close()

# Load comments from file
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_text():
  # Make text all lowercase and split into sentences
  # Load in new raw here
  raw = get_comment_set(200)
  raw = raw.lower()
  raw_len = len(raw)
  raw = tokenize.sent_tokenize(raw)
  print(len(raw))
  return raw

# Generate overlapping sequences of words
def get_sequences(raw):
    sequences = []
    maxLen = 0
    for sequence in raw:
        token_list = word_tokenize(sequence)
        token_list = [token for token in token_list if token.isalpha()]
        i = 0
        while (i < (len(token_list)-1)):
            tokens = token_list[:i+1]
            line = ' '.join(tokens)
            sequences.append(line)
            i = i + 1
    filename = 'reddit_comments.txt'
    save_doc(sequences, filename)
    return sequences

# Map our words to integer values and split sequences into input and labels
def prepare_sequences(sequences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    maxLen = max([len(x) for x in sequences])
    num_vocab = len(tokenizer.word_index) + 1
    input_sequences = np.array(pad_sequences(sequences, maxlen = maxLen-1, padding = 'pre'))
    x = input_sequences[:,:-1]
    y = input_sequences[:,-1]
    y = to_categorical(y, num_classes=num_vocab)
    len_sequence = x.shape[1]
    # save the tokenizer
    dump(tokenizer, open('tokenizer.pkl', 'wb'))
    return x, y, len_sequence, num_vocab



