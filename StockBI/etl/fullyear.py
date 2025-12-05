import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
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
test_days = 30
train = data.iloc[:-test_days].copy()
test = data.iloc[-test_days:].copy()

# Ensure at least 1 year of data in train (if available)
if len(train) > 252:  # about 252 trading days in a year
    train = train.iloc[-252:]

# ----------------------------
# ARIMA Forecast
# ----------------------------
arima_preds = []
history = list(train["Close"])
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast()[0]
    arima_preds.append(forecast)
    history.append(test["Close"].iloc[t])  # update history iteratively

arima_forecast = pd.Series(arima_preds, index=test.index)

# ----------------------------
# Random Forest Forecast
# ----------------------------
lags = 5
rf_data = train.copy()

# create lag features
for lag in range(1, lags + 1):
    rf_data[f"lag_{lag}"] = rf_data["Close"].shift(lag)
rf_data = rf_data.dropna()

X_train = rf_data[[f"lag_{i}" for i in range(1, lags + 1)]]
y_train = rf_data["Close"]

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# iterative forecast
rf_preds = []
history = list(train["Close"][-lags:])  # last 'lags' points

for _ in range(len(test)):
    features = np.array(history[-lags:]).reshape(1, -1)
    pred = rf_model.predict(features)[0]
    rf_preds.append(pred)
    history.append(test["Close"].iloc[_])  # use actual for next step

rf_forecast = pd.Series(rf_preds, index=test.index)

# ----------------------------
# Prophet Forecast
# ----------------------------
prophet_preds = []
history = train[["Date", "Close"]].copy()
for t in range(len(test)):
    df_prophet = history.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    pred = forecast["yhat"].iloc[-1]
    prophet_preds.append(pred)
    next_row = pd.DataFrame({"Date": [test["Date"].iloc[t]], "Close": [test["Close"].iloc[t]]})
    history = pd.concat([history, next_row], ignore_index=True)

prophet_forecast = pd.Series(prophet_preds, index=test.index)

# ----------------------------
# Plot Comparison
# ----------------------------
plt.figure(figsize=(12, 6))
plt.scatter(test["Date"], test["Close"], color="black", label="Actual", s=30)
plt.plot(test["Date"], arima_forecast, color="red", label="ARIMA Forecast")
plt.plot(test["Date"], rf_forecast, color="green", label="Random Forest Forecast")
plt.plot(test["Date"], prophet_forecast, color="blue", label="Prophet Forecast")
plt.title(f"{ticker} Close Price Forecast (1-day ahead using 1 year of training data)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()
