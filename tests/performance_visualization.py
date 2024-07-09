# Imports
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/Mohamed/OneDrive/Documents/GitHub/stock-predictor/data')
from data_collection import get_data
sys.path.insert(0, '/Users/Mohamed/OneDrive/Documents/GitHub/stock-predictor/src')
from predictor_model import predict_stock

# Training/Test data initialization
df, train_data, test_data, scaler = get_data()
predicted_stock_price = predict_stock()

plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data['Date'], train_data['Close'], label='Training Data')
plt.plot(test_data['Date'], test_data['Close'], color = 'blue', label='Actual Stock Price')
plt.plot(test_data[60:]['Date'], predicted_stock_price, color = 'orange', label="Predicted Stock Price")

plt.title("Tesla Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Tesla Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()