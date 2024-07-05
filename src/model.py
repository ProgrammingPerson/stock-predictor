# Imports
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import sys
sys.path.insert(0, '/Users/Mohamed/OneDrive/Documents/GitHub/stock-predictor/data')
from data_prep import getTrainingX

# Training data initialization
x_train, y_train = getTrainingX()

# Model initialization
model = Sequential()

# LSTM Layer 1
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.25))

# LSTM Layer 2
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.25))

# LSTM Layer 3
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.25))

# LSTM Layer 2
model.add(LSTM(units = 50))
model.add(Dropout(0.25))

# Final Layer
model.add(Dense(units = 1))
model.summary()