from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Imports 
import sys



model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))