import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Fetch Stock Data
stock = 'AAPL'

data = yf.download(stock, start='2015-01-01', end='2025-01-01')

print(data.head())

# Plot Closing Price
plt.figure(figsize=(12,6))
plt.plot(data['Close'])
plt.title('Apple Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Prepare Dataset
dataset = data['Close'].values
dataset = dataset.reshape(-1,1)

# Normalize Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create Training Data
x_train = []
y_train = []

for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i,0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape Data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM Model
model = Sequential()

model.add(LSTM(units=50,
               return_sequences=True,
               input_shape=(x_train.shape[1],1)))

model.add(Dropout(0.2))

model.add(LSTM(units=50,
               return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(units=50))

model.add(Dropout(0.2))

model.add(Dense(units=1))

# Compile Model
model.compile(optimizer='adam',
              loss='mean_squared_error')

# Train Model
model.fit(x_train,
          y_train,
          epochs=10,
          batch_size=32)

print("Model Trained Successfully")

# Test Data
test_data = yf.download(stock,
                        start='2024-01-01',
                        end='2025-01-01')

actual_prices = test_data['Close'].values

# Combine datasets
total_dataset = pd.concat((data['Close'],
                           test_data['Close']),
                           axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - 60:].values

model_inputs = model_inputs.reshape(-1,1)

model_inputs = scaler.transform(model_inputs)

# Create Test Sequences
x_test = []

for i in range(60, len(model_inputs)):
    x_test.append(model_inputs[i-60:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test,
                    (x_test.shape[0],
                     x_test.shape[1],
                     1))

# Predict Prices
predicted_prices = model.predict(x_test)

predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot Results
plt.figure(figsize=(12,6))

plt.plot(actual_prices,
         color='black',
         label='Actual Price')

plt.plot(predicted_prices,
         color='green',
         label='Predicted Price')

plt.title('Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Price')

plt.legend()

plt.show()

# Moving Average
data['MA50'] = data['Close'].rolling(window=50).mean()

plt.figure(figsize=(12,6))

plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA50'], label='50-Day Moving Average')

plt.title('Moving Average Indicator')

plt.xlabel('Date')
plt.ylabel('Price')

plt.legend()

plt.show()

# RSI Indicator
delta = data['Close'].diff()

gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()

loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

rs = gain / loss

data['RSI'] = 100 - (100 / (1 + rs))

# Plot RSI
plt.figure(figsize=(12,6))

plt.plot(data['RSI'])

plt.title('RSI Indicator')

plt.xlabel('Date')
plt.ylabel('RSI')

plt.show()