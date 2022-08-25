from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
import pandas as pd

data = pd.read_csv("data/OnionOrNot.csv")

print(data)