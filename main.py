from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, Flatten
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("data/OnionOrNot.csv")
#Our Dataset

data = data.sample(frac=1) #randomizes data

tokens = Tokenizer() #tokenizes dataset
tokens.fit_on_texts(data.text.values)

def remove_high_freq(l, thresh):
    return([i - thresh if i > thresh else 0 for i in l ])

def remove_low_freq(l, thresh):
    return([i for i in l if i < thresh])

vocabulary = len(tokens.word_index)
print(vocabulary)

x = tokens.texts_to_sequences(data.text.values)
x = pad_sequences(x) #pads lengths of x out to longest list
y = data.label

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
X_train, X_val,  y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)


model = Sequential()
model.add(Embedding(vocabulary+1, 64, input_length = X_train.shape[1])) #input_length tells model the shape of the data
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 32, epochs = 5, validation_data = (X_val, y_val))
model.summary()

results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)