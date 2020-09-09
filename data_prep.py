from __future__ import division
import nltk, re, pprint
from nltk import tokenize
from nltk import word_tokenize
from urllib import request
import numpy as np

# Load in Jane Eyre from Project Gutenberg 
url = "http://www.gutenberg.org/cache/epub/1260/pg1260.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
# Remove unwanted text
start = raw.find("CHAPTER I")
end = raw.rfind("***END OF THE PROJECT GUTENBERG")
raw = raw[start:end]

# Get set of unique characters as our features
raw = raw.lower()
raw_len = len(raw)
#raw = raw.split()
features = sorted(list(set(raw)))
#print(features)
num_features = len(features)
feature_index = dict((f, i) for i, f in enumerate(features))
raw = tokenize.sent_tokenize(raw)

# Length of character sequences, chunked out in steps of 3
maxLen = 40
step = 3
sequences = []
next_char = []

# Generate overlapping sequences of words
for sequence in raw:
    token_list = word_tokenize(sequence)
    i = 0
    while (i < len(token_list)):
        n_gram = token_list[:i+1]
        sequences.append(n_gram)
        i = i + 1
print(sequences)
x = np.zeros((len(sequences), maxLen, num_features), dtype=np.bool)
y = np.zeros((len(sequences), num_features), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for j, char in enumerate(sequence):
        x[i, j, feature_index[char]] = 1
    y[i, feature_index[next_char[i]]] = 1





