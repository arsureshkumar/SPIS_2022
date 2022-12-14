from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, Flatten
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("data/OnionOrNot.csv")
#Our Dataset

#data preprocessing
tokens = Tokenizer() #tokenizes dataset
tokens.fit_on_texts(data.text.values)
vocabulary = len(tokens.word_index)
x = tokens.texts_to_sequences(data.text.values)
x = pad_sequences(x) #pads lengths of x out to longest list
y = data.label

#split data into training, test, and validation
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
X_train, X_val,  y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

#build model
model = Sequential()
model.add(Embedding(vocabulary+1, 64, input_length = X_train.shape[1])) #input_length tells model the shape of the data
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fit model
model.fit(X_train, y_train, batch_size = 32, epochs = 5, validation_data = (X_val, y_val))

#evaluate model
model.summary()
print("test loss, test acc:", model.evaluate(X_test, y_test, batch_size=128))