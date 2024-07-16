# Imports
import numpy as np
from data_collection import get_data

def get_training():
    df, train_data, test_data, scaler = get_data()
    x_train=[]
    y_train = []
    tickers = []

    # Ticker input
    ticker_input = open("tickers.txt", "r")
    for _ in range(52):
        tickers.append(ticker_input.readline().strip())

    for ticker in tickers:
        # 60 day look back for the training model
        for i in range(60, len(train_data[ticker]['scaled_values'])):
            y_train.append(train_data[ticker]['scaled_values'][i])
            x_train.append(train_data[ticker]['scaled_values'][i-60:i])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # LSTM model as 3d array
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    return x_train, y_train

def get_test(ticker):
    df, train_data, test_data, scaler = get_data()

    x_test=[]
    y_test = test_data[ticker]['scaled_values']

    for i in range(60, len(test_data[ticker])):
        x_test.append(test_data[ticker]['scaled_values'][i-60:i])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    return x_test, y_test, scaler