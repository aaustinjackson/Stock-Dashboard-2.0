import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import os

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
# Parameters
# ----------------------------
window = 3  # 3 prior days for prediction
lags = 3    # for Random Forest

# ----------------------------
# ARIMA Forecast
# ----------------------------
arima_preds = []
history = list(train["Close"])
for t in range(len(test)):
    model = ARIMA(history[-window:], order=(1, 0, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast()[0]
    arima_preds.append(forecast)
    history.append(test["Close"].iloc[t])

arima_forecast = pd.Series(arima_preds, index=test.index)

# ----------------------------
# Random Forest Forecast
# ----------------------------
rf_data = train.copy()
# Create lag features
for lag in range(1, lags + 1):
    rf_data[f"lag_{lag}"] = rf_data["Close"].shift(lag)
rf_data = rf_data.dropna()

X_train = rf_data[[f"lag_{i}" for i in range(1, lags + 1)]]
y_train = rf_data["Close"]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Iterative forecast
rf_preds = []
history = list(train["Close"][-lags:])

for _ in range(len(test)):
    features = pd.DataFrame([history[-lags:]], columns=[f"lag_{i}" for i in range(1, lags + 1)])
    pred = rf_model.predict(features)[0]
    rf_preds.append(pred)
    history.append(test["Close"].iloc[_])

rf_forecast = pd.Series(rf_preds, index=test.index)

# ----------------------------
# Prophet Forecast (fit once)
# ----------------------------
prophet_train = train.reset_index().rename(columns={"Date": "ds", "Close": "y"})
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_train)

future = prophet_model.make_future_dataframe(periods=len(test))
forecast = prophet_model.predict(future)
prophet_forecast = forecast.set_index("ds")["yhat"].iloc[-len(test):]
prophet_forecast.index = test.index  # align index with test set

# ----------------------------
# Plot forecasts
# ----------------------------
plt.figure(figsize=(12, 6))
plt.scatter(test.index, test["Close"], color="black", label="Actual", s=30)
plt.plot(test.index, arima_forecast, color="red", label="ARIMA Forecast")
plt.plot(test.index, rf_forecast, color="green", label="Random Forest Forecast")
plt.plot(test.index, prophet_forecast, color="blue", label="Prophet Forecast")
plt.title(f"{ticker} Close Price Forecast (1-day ahead using last {window} days)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Compute forecast errors
# ----------------------------
arima_error = pd.Series(test["Close"].values - arima_forecast.values, index=test.index)
rf_error = pd.Series(test["Close"].values - rf_forecast.values, index=test.index)
prophet_error = pd.Series(test["Close"].values - prophet_forecast.values, index=test.index)

# ----------------------------
# Plot forecast errors
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(test.index, arima_error, color="red", label="ARIMA Error")
plt.plot(test.index, rf_error, color="green", label="Random Forest Error")
plt.plot(test.index, prophet_error, color="blue", label="Prophet Error")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"{ticker} Forecast Errors (Actual - Predicted)")
plt.xlabel("Date")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()
