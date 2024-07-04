import matplotlib.pyplot as plt
from data_collection import get_data

df, train_data, test_data = get_data()

# Initial graph
plt.plot(df["Date"],df["Close"])
plt.title("Tesla stock price vs Time")
plt.xlabel("time")
plt.ylabel("price")

# Graph of test/training split
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df['Date'], df['Close'], 'green', label='Train data')
plt.plot(test_data['Date'], test_data['Close'], 'blue', label='Test data')
plt.legend()
plt.show()
