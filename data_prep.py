from __future__ import division
import nltk, re, pprint
from nltk import tokenize
from nltk import word_tokenize
from urllib import request
import numpy as np
import string

# Load in Jane Eyre from Project Gutenberg 
url = "http://www.gutenberg.org/files/58585/58585-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
# Remove unwanted text
start = raw.find("Almustafa")
end = raw.rfind("*** END OF THIS PROJECT")
raw = raw[start:end]

# Get set of unique characters as our features
raw = raw.lower()
raw_len = len(raw)
tokens = word_tokenize(raw)
words = [word for word in tokens if word.isalpha()]
#features = sorted(list(set(word_tokenize(raw))))
features = sorted(list(set(words)))
num_features = len(features)
print(num_features)
feature_index = dict((f, i) for i, f in enumerate(features))
raw = tokenize.sent_tokenize(raw)

# Generate overlapping sequences of words
sequences = []
next_word = []
maxLen = 0
for sequence in raw:
    token_list = word_tokenize(sequence)
    token_list = [token for token in token_list if token.isalpha()]
    i = 0
    if (len(token_list) > maxLen):
        maxLen = len(token_list)
    while (i < (len(token_list)-1)):
        n_gram = token_list[:i+1]
        sequences.append(n_gram)
        next_word.append(token_list[i+1])
        i = i + 1
print(sequences)
x = np.zeros((len(sequences), maxLen, num_features), dtype=np.bool)
y = np.zeros((len(sequences), num_features), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for j, char in enumerate(sequence):
        x[i, j, feature_index[char]] = 1
    y[i, feature_index[next_word[i]]] = 1





