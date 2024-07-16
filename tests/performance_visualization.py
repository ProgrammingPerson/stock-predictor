# Imports
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/Mohamed/OneDrive/Documents/GitHub/stock-predictor/data')
from data_collection import get_data
sys.path.insert(0, '/Users/Mohamed/OneDrive/Documents/GitHub/stock-predictor/src')
from predictor_model import predict_stock, create_model, load_model

# Model initialization
model = create_model()
load_model(model, "training/cp.weights.h5")

# User input for stock to evaluate
ticker = str(input("Please enter the ticker of the stock to predict: ")).strip()

# Training/Test data initialization
df, train_data, test_data, scaler = get_data()
predicted_stock_price = predict_stock(model, ticker)

plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data[ticker]['Date'], train_data[ticker]['Close'], label='Training Data')
plt.plot(test_data[ticker]['Date'], test_data[ticker]['Close'], color = 'blue', label='Actual Stock Price')
plt.plot(test_data[ticker][60:]['Date'], predicted_stock_price, color = 'orange', label="Predicted Stock Price")

plt.title("Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()