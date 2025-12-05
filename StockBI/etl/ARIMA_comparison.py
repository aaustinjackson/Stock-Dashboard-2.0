import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# Paths
# ----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "top10_stock_data.csv")

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(data_path, parse_dates=["Date"])
ticker = "AAPL"
df = df[df["Ticker"] == ticker].sort_values("Date")

data = df[["Date", "Close"]].copy()
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data = data.dropna(subset=["Close"])

# ----------------------------
# Train/test split
# ----------------------------
train = data.iloc[:-30].copy()
test = data.iloc[-30:].copy()
train.set_index("Date", inplace=True)
test.set_index("Date", inplace=True)

# ----------------------------
# Function for ARIMA rolling forecast
# ----------------------------
def arima_forecast(train_series, test_series, window_size):
    history = list(train_series)
    preds = []

    for t in range(len(test_series)):
        window_data = history[-window_size:]

        if len(window_data) < 3:  # too few to fit ARIMA
            yhat = window_data[-1]  # fallback: use last value
        else:
            try:
                model = ARIMA(window_data, order=(1, 0, 0))
                model_fit = model.fit()
                yhat = model_fit.forecast()[0]
            except:
                yhat = window_data[-1]  # fallback if ARIMA fails

        preds.append(yhat)
        history.append(test_series.iloc[t])

    return pd.Series(preds, index=test_series.index)

# ----------------------------
# Run ARIMA with 3 different window sizes
# ----------------------------
arima_1 = arima_forecast(train["Close"], test["Close"], window_size=1)
arima_3 = arima_forecast(train["Close"], test["Close"], window_size=3)
arima_5 = arima_forecast(train["Close"], test["Close"], window_size=5)

# ----------------------------
# Plot the results
# ----------------------------
plt.figure(figsize=(12, 6))
plt.scatter(test.index, test["Close"], color="black", label="Actual", s=30)
plt.plot(test.index, arima_1, color="red", label="ARIMA (1-day lag)")
plt.plot(test.index, arima_3, color="blue", label="ARIMA (3-day lag)")
plt.plot(test.index, arima_5, color="green", label="ARIMA (5-day lag)")
plt.title(f"{ticker} ARIMA Forecasts with Different Window Sizes")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()
