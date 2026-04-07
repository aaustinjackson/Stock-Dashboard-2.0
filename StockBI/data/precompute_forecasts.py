import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

# ---------------------------------------------
# Paths
# ---------------------------------------------
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(repo_root, "data", "top10_stock_data_cleaned.csv")
output_dir = os.path.join(repo_root, "data", "precomputed_forecasts")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------
# Load data
# ---------------------------------------------
df_all = pd.read_csv(data_path, parse_dates=["Date"])
df_all["Close"] = pd.to_numeric(df_all["Close"], errors="coerce")
df_all = df_all.dropna(subset=["Close"])
tickers = df_all["Ticker"].dropna().astype(str).unique()

# ---------------------------------------------
# ARIMA (walk-forward, uses actuals)
# ---------------------------------------------
def forecast_arima(df, window=5):
    history = []
    preds = []

    for i in range(len(df)):
        actual = df["Close"].iloc[i]
        if i < window:
            preds.append(actual)
            history.append(actual)
            continue

        window_data = history[-window:]
        try:
            model = ARIMA(window_data, order=(1, 1, 0))
            pred = model.fit().forecast()[0]
        except:
            pred = window_data[-1]

        preds.append(pred)
        history.append(actual)  # always use actual to avoid drift

    return pd.Series(preds, index=df["Date"])

# ---------------------------------------------
# Random Forest (train once, predict with rolling actuals)
# ---------------------------------------------
def forecast_rf(df, lags=10):
    closes = df["Close"].astype(float).values
    dates = df["Date"].values
    n = len(df)

    # Build lag features from the full series
    def make_features(close_array):
        rows = []
        for i in range(len(close_array)):
            if i < lags:
                rows.append(None)
                continue
            lag_vals = list(close_array[i - lags:i])
            rows.append(lag_vals)
        return rows

    feature_rows = make_features(closes)

    # Collect valid training samples (all rows where we have lags)
    X_train, y_train = [], []
    for i in range(lags, n):
        X_train.append(feature_rows[i])
        y_train.append(closes[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Predict every row using a rolling window of actuals
    preds = []
    history = list(closes)  # use full actuals for rolling window

    for i in range(n):
        if i < lags:
            preds.append(closes[i])  # not enough history, use actual
            continue
        features = np.array(history[i - lags:i]).reshape(1, -1)
        pred = model.predict(features)[0]
        preds.append(pred)

    return pd.Series(preds, index=df["Date"])

# ---------------------------------------------
# Prophet
# ---------------------------------------------
def forecast_prophet(df):
    if len(df) < 2:
        return pd.Series(df["Close"].values, index=df["Date"])

    history = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05
    )
    model.fit(history)
    forecast = model.predict(history)
    return pd.Series(forecast["yhat"].values, index=df["Date"])

# ---------------------------------------------
# Run forecasts for all tickers
# ---------------------------------------------
for ticker in tickers:
    df_ticker = (
        df_all[df_all["Ticker"] == ticker]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(f"Computing {ticker} ({len(df_ticker)} rows)...")

    arima_preds = forecast_arima(df_ticker)
    rf_preds = forecast_rf(df_ticker)
    prophet_preds = forecast_prophet(df_ticker)

    combined = pd.DataFrame({
        "Date": df_ticker["Date"],
        "Actual": df_ticker["Close"],
        "ARIMA": arima_preds.values,
        "RF": rf_preds.values,
        "Prophet": prophet_preds.values
    })

    output_file = os.path.join(output_dir, f"{ticker}_forecasts.csv")
    if ticker == "GOOG":
        output_file = os.path.join(output_dir, "GOOGL_forecasts.csv")

    combined.to_csv(output_file, index=False)
    print(f"  Saved → {output_file}")

print("✅ Done.")
