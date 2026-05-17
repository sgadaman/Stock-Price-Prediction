import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.title("Stock Price Dashboard")

stock = st.text_input("Enter Stock Symbol", "AAPL")

data = yf.download(stock, start="2020-01-01", end="2025-01-01")

st.subheader("Stock Data")
st.write(data.tail())

# Closing Price Graph
st.subheader("Closing Price Graph")

fig = plt.figure(figsize=(12,6))

plt.plot(data['Close'])

plt.xlabel("Date")
plt.ylabel("Price")

st.pyplot(fig)


# Moving Average
data['MA50'] = data['Close'].rolling(window=50).mean()

st.subheader("50-Day Moving Average")

fig2 = plt.figure(figsize=(12,6))

plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA50'], label='MA50')

plt.legend()

st.pyplot(fig2)

# RSI Indicator
delta = data['Close'].diff()

gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()

loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

rs = gain / loss

data['RSI'] = 100 - (100 / (1 + rs))

st.subheader("RSI Indicator")

fig3 = plt.figure(figsize=(12,6))

plt.plot(data['RSI'])

plt.axhline(70, linestyle='--')

plt.axhline(30, linestyle='--')

st.pyplot(fig3)
