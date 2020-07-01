from __future__ import division
import nltk, re, pprint
from nltk import word_tokenize
from urllib import request

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
features = sorted(list(set(raw)))
num_features = len(features)
feature_index = dict((f, i) for i, f in enumerate(features))

# Length of character sequences, chunked out in steps of 3
maxLen = 40
step = 3
sequences = []
next_char = []
for i in range(0, raw_len - maxLen, step):
    sequences.append(raw[i:i + maxLen])
    next_char.append(raw[i + maxLen])





