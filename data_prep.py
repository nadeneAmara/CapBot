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
print(type(raw))
