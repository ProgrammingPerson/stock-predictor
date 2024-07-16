# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yfinance as yf
import requests_cache as rc
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
import requests
import warnings
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
import requests_ratelimiter
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

warnings.filterwarnings('ignore')

def get_tickers():
    url = 'https://www.marketbeat.com/stocks/sectors/computer-and-technology/#:~:text=Computer%20and%20Technology%20Stocks%20List%201%20%231%20-,Semiconductor%20Manufacturing%20NYSE%3ATSM%20Stock%20Price%3A%20%24184.47%20%28-%246.58%29%20'

    headers={"User-Agent": "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"}

    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    tickers = []

    for ticker in soup.find_all('a'):
        if "VMW" in ticker.text:
            continue
        elif "NASDAQ:" in ticker.text:
            tickers.append(ticker.text[7:])
        elif "NYSE:" in ticker.text:
            tickers.append(ticker.text[5:])

    # Ticker output file
    output = open("tickers.txt", "w")
    for ticker in tickers:
        output.write(ticker + "\n")
    output.close()

def get_data():
# API call caching and rate-limiting
    session = rc.CachedSession('yfinance.cache')
    session.headers['User-agent'] = 'my-program/1.0'
    session = CachedLimiterSession(limiter=Limiter(RequestRate(1,Duration.SECOND*1)),
                                bucket_class=MemoryQueueBucket,
                                backend=SQLiteCache("yfinance.cache"))
    
    # Data dict initializaiton
    train_data = {}
    test_data = {}

    # Ticker input
    ticker_file = open("tickers.txt", "r")
    for _ in range(52):
        ticker = ticker_file.readline().strip()

        # Data downloading
        df = yf.download(ticker, session=session).reset_index()
        df = df[(df['Date'] >= "2012-01-01") & (df['Date'] <= "2024-06-06")].reset_index(drop=True)

        # Data normalization
        scaler = MinMaxScaler(feature_range=(0,1))
        df['scaled_values'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        # Training and test data split
        train_data[ticker] = df[df['Date']<'2023-01-01']
        test_data[ticker] = df[df['Date']>='2023-01-01']

    ticker_file.close()
    
    return df, train_data, test_data, scaler