from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Conv1D
import pandas as pd
from sklearn.model_selection import train_test_split

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



X_train, X_test, y_train, y_test = train_test_split(data.text, data.label, test_size = 0.2)
X_train, X_val,  y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

print(X_train)

vocabulary -= 100
vocabulary -= 3000

model = Sequential()
model.add(Embedding(vocabulary, 32))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 32, epochs = 2, validation_data = (X_val, y_val))

results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)

print("hello")