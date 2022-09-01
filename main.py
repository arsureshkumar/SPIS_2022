from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, Flatten
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#data = pd.read_csv("data/OnionOrNot.csv")
#First Dataset, to use un-comment line 9 and comment lines 22-23

#Convert True and False datasets to pandas, modify, then combine
tdata = pd.read_csv("data/True.csv") #True data
fdata = pd.read_csv("data/Fake.csv") #False data

tdata["label"] = 0 #adds label 0 to true data
fdata["label"] = 1 #adds label 1 to false data

for i in [tdata, fdata]:
    i["text"] = i["title"].astype(str) + " " + i["body"] #Creates text column which is ombination of title and body
print(fdata)
#data = [i.drop(['subject', 'title', 'body'], axis=1) for i in [tdata, fdata]] #drops unecessary data
concatdata = [tdata, fdata]
data = pd.concat(concatdata) #combines true and false datasets into data

data = data.sample(frac=1) #randomizes data


tokens = Tokenizer()
tokens.fit_on_texts(data.text.values)

def remove_high_freq(l, thresh):
    return([i - thresh if i > thresh else 0 for i in l ])

def remove_low_freq(l, thresh):
    return([i for i in l if i < thresh])

vocabulary = len(tokens.word_index)
print(vocabulary)

#data.text = data.text.map(lambda x: remove_high_freq(x, 100))
#data.text = data.text.map(lambda x: remove_low_freq(x, vocabulary-3000))
#x = np.asarray(list(data.text)).astype('float32')
#y = np.asarray(list(data.label)).astype('float32')

x = tokens.texts_to_sequences(data.text.values) #instead of writing "data.text = ...", write "x = ... and use .values"
x = pad_sequences(x) #replaces add_zeroes, pads lengths of x out to longest list
y = data.label

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
X_train, X_val,  y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)


model = Sequential()
model.add(Embedding(vocabulary+1, 64, input_length = X_train.shape[1])) #input_length tells model the shape of the data, When 1 isnt added to vocabulary it always throws an out of bounds error so that was my short term fix lol
model.add(Conv1D(filters=32, kernel_size=8, activation='relu')) #Code works with or without, I cant tell if it changes anything
#model.add(Conv1D(filters=32, kernel_size=8, activation='relu')) # added another convolutional layer to improve context detection
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 32, epochs = 5, validation_data = (X_val, y_val))
model.summary()

results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)