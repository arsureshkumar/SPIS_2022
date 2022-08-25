from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
import pandas as pd

data = pd.read_csv("data/OnionOrNot.csv")

tokens = Tokenizer()
tokens.fit_on_texts(data.text)
#print(tokens.word_counts)
#print(tokens.word_index)

def remove_high_freq(l, thresh):
    return([i for i in l if i > thresh])



data.text = tokens.texts_to_sequences(data.text)
data.text = data.text.map(lambda x: remove_high_freq(x, 100))

print(len(tokens.word_index))
print(data)