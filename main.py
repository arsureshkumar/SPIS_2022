from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#data = pd.read_csv("data/OnionOrNot.csv")
#First Dataset, to use un-comment line 8 and comment lines 30-33


#Convert True and False datasets to pandas, modify, then combine
tdata = pd.read_csv("data/True.csv") #True data
fdata = pd.read_csv("data/Fake.csv") #False data

tdata["label"] = 0 #adds label 0 to true data
fdata["label"] = 1 #adds label 1 to false data

for i in [tdata, fdata]:
    i["text"] = i["title"].astype(str) + " " + i["body"] #Creates text column which is ombination of title and body

data = [i.drop(['subject', 'title', 'body'], axis=1) for i in [tdata, fdata]] #drops unecessary data

data = pd.concat(data, ignore_index = True) #combines true and false datasets into data
data = data.sample(frac=1) #randomizes data


tokens = Tokenizer()
tokens.fit_on_texts(data.text)

def remove_high_freq(l, thresh):
    return([i - thresh if i > thresh else 0 for i in l ])

def remove_low_freq(l, thresh):
    return([i for i in l if i < thresh])

vocabulary = len(tokens.word_index)
print(vocabulary)

data.text = tokens.texts_to_sequences(data.text)
#data.text = data.text.map(lambda x: remove_high_freq(x, 100))
#data.text = data.text.map(lambda x: remove_low_freq(x, vocabulary-3000))

print(data)


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


model = Sequential()
model.add(Embedding(vocabulary, 64))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu')) # added another convolutional layer to improve context detection
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 32, epochs = 5, validation_data = (X_val, y_val))

results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)