from __future__ import division
import string
import nltk, re, pprint
from nltk import tokenize
from nltk import word_tokenize
from urllib import request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

def load_text(url):
    response = request.urlopen(url)
    raw = response.read().decode('utf8')
    # Remove unwanted text
    start = raw.find("Almustafa")
    end = raw.rfind("*** END OF THIS PROJECT")
    raw = raw[start:end]
    # Make text all lowercase and split into sentences
    raw = raw.lower()
    raw_len = len(raw)
    raw = tokenize.sent_tokenize(raw)
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
    return sequences

# Map our words to integer values and split sequences into 
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
    return x, y, maxLen, num_vocab







