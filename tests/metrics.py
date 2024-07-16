import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from predictor_model import predict_stock
import sys
sys.path.insert(0, '/Users/Mohamed/OneDrive/Documents/GitHub/stock-predictor/data')
from data_collection import get_data

df, train_data, test_data, scaler = get_data()

y_true = test_data[60:]['Close'].values
y_pred = predict_stock()

mse = mean_squared_error(y_true, y+pred)
print('MSE: '+str(mse))
mae = mean_absolute_error(y_true,y_pred)
print('MAE: '+ str(mae))
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
print('RMSE: '+ str(rmse))
mape = np.mean(np.abs(y_pred-y_true)/np.abs(y_true))
print('MAPE: '+str(mape))