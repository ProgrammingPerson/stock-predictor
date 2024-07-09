# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

def get_data():
# Data downloading
    df = yf.download('TSLA').reset_index()
    df = df[(df['Date'] >= "2012-01-01") & (df['Date'] <= "2024-06-06")].reset_index(drop=True)

    # Data normalization
    scaler = MinMaxScaler(feature_range=(0,1))
    df['scaled_values'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Training and test data split
    train_data = df[df['Date']<'2023-01-01']
    test_data = df[df['Date']>='2023-01-01']
    
    return df, train_data, test_data, scaler