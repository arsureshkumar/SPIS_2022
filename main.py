from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#data = pd.read_csv("data/OnionOrNot.csv")
tdata = pd.read_csv("data/True.csv") #True data
fdata = pd.read_csv("data/Fake.csv") #False data

tdata["text"] = tdata["title"].astype(str) + " " + tdata["body"]
tdata["label"] = "0"
tdata.pop('subject')
tdata.pop('title')
tdata.pop('body')

fdata["text"] = fdata["title"].astype(str) + " " + fdata["body"]
fdata["label"] = "1"
fdata.pop('subject')
fdata.pop('title')
fdata.pop('body')

datalist = [tdata, fdata]
data = pd.concat(datalist)
data = data.sample(frac=1)

tokens = Tokenizer()
tokens.fit_on_texts(data.text)

def remove_high_freq(l, thresh):
    return([i - thresh if i > thresh else 0 for i in l ])

def remove_low_freq(l, thresh):
    return([i for i in l if i < thresh])

vocabulary = len(tokens.word_index)
print(vocabulary)

data.text = tokens.texts_to_sequences(data.text)
data.text = data.text.map(lambda x: remove_high_freq(x, 1000))
data.text = data.text.map(lambda x: remove_low_freq(x, vocabulary-30000))

maximum_length = len(max(data.text, key=len))
def add_zeroes(l):
    while len(l) < maximum_length:
        l.append(0)
    return(l)
data.text = data.text.map(add_zeroes)


x = np.asarray(list(data.text)).astype('float32')
y = np.asarray(list(data.label)).astype('float32')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
X_train, X_val,  y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

print(X_train)



model = Sequential()
model.add(Embedding(vocabulary, 32))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 32, epochs = 2, validation_data = (X_val, y_val))

results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)
