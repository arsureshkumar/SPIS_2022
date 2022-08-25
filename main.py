from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
import pandas as pd

data = pd.read_csv("data/OnionOrNot.csv")

tokens = Tokenizer()
# fit the tokenizer on the documents
tokens.fit_on_texts(data.text)
# summarize what was learned
print(tokens.word_counts)
print(tokens.document_count)
print(tokens.word_index)
print(tokens.word_docs)

print(data)