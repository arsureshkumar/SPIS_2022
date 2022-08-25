from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
import pandas as pd

data = pd.read_csv("data/OnionOrNot.csv")

tokens = Tokenizer()
tokens.fit_on_texts(data.text)

def remove_high_freq(l, thresh):
    return([i for i in l if i > thresh])

def remove_low_freq(l, thresh):
    return([i for i in l if i < thresh])

vocabulary = len(tokens.word_index)

data.text = tokens.texts_to_sequences(data.text)
data.text = data.text.map(lambda x: remove_high_freq(x, 100))
data.text = data.text.map(lambda x: remove_low_freq(x, vocabulary-3000))

vocabulary -= 100
vocabulary -= 3000

model = Sequential()
model.add(Embedding(vocabulary, 32))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.

print(data)