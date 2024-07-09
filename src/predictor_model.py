# Imports
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import sys
sys.path.insert(0, '/Users/Mohamed/OneDrive/Documents/GitHub/stock-predictor/data')
from data_prep import get_training, get_test
from sklearn.preprocessing import MinMaxScaler

# Training/Test data initialization
x_train, y_train = get_training()
x_test, y_test, scaler = get_test()

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

# Model Compilation
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Model Training
model.fit(x_train, y_train, epochs = 10, batch_size = 32)

def predict_stock():
    # Prediction on test data
    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    return predicted_stock_price