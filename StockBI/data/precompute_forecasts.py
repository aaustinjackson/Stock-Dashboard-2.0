import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

# ---------------------------------------------
# Paths
# ---------------------------------------------
data_path = r"C:\Users\austi\PycharmProjects\StockBI\data\top10_stock_data_cleaned.csv"
output_dir = r"C:\Users\austi\PycharmProjects\StockBI\data\precomputed_forecasts"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------
# Load stock data
# ---------------------------------------------
df_all = pd.read_csv(data_path, parse_dates=["Date"])
df_all["Close"] = pd.to_numeric(df_all["Close"], errors="coerce")
tickers = df_all["Ticker"].dropna().astype(str).unique()

# ---------------------------------------------
# Forecast functions
# ---------------------------------------------
def forecast_arima(train_df, test_df, window=3):
    history = list(train_df["Close"].astype(float))
    preds = []
    for i in range(len(test_df)):
        window_data = history[-window:] if len(history) >= window else history
        try:
            model = ARIMA(window_data, order=(1,1,0))
            model_fit = model.fit()
            pred = model_fit.forecast()[0]
        except:
            pred = window_data[-1]
        preds.append(pred)
        history.append(float(test_df["Close"].iloc[i]) if pd.notna(test_df["Close"].iloc[i]) else float(pred))
    return pd.Series(preds, index=test_df["Date"])

def forecast_rf(train_df, test_df, lags=3):
    if train_df.empty:
        val = float(test_df["Close"].iloc[0] if pd.notna(test_df["Close"].iloc[0]) else 0)
        return pd.Series([val]*len(test_df), index=test_df["Date"])
    rf_data = train_df.copy().sort_values("Date")
    for lag in range(1, lags+1):
        rf_data[f"lag_{lag}"] = rf_data["Close"].shift(lag)
    rf_data = rf_data.dropna()
    if rf_data.empty:
        val = float(train_df["Close"].iloc[-1])
        return pd.Series([val]*len(test_df), index=test_df["Date"])
    X_train = rf_data[[f"lag_{i}" for i in range(1,lags+1)]].values
    y_train = rf_data["Close"].values
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    history_vals = list(train_df["Close"].astype(float).values[-lags:])
    preds = []
    for i in range(len(test_df)):
        features = np.array(history_vals[-lags:]).reshape(1,-1)
        pred = model.predict(features)[0]
        preds.append(pred)
        val = test_df["Close"].iloc[i]
        history_vals.append(float(val) if pd.notna(val) else float(pred))
    return pd.Series(preds, index=test_df["Date"])

def forecast_prophet(train_df, test_df):
    if len(train_df) < 2:
        val = train_df["Close"].iloc[-1] if len(train_df) > 0 else 0
        return pd.Series([val]*len(test_df), index=test_df["Date"])
    history = train_df[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, seasonality_mode="additive",
                    changepoint_prior_scale=0.05)
    model.fit(history)
    future = model.make_future_dataframe(periods=len(test_df), freq='D')
    forecast = model.predict(future)
    forecast_series = forecast.set_index("ds")["yhat"]
    aligned = forecast_series.reindex(pd.to_datetime(test_df["Date"]), method="nearest")
    aligned.index = pd.to_datetime(test_df["Date"])
    return aligned

# ---------------------------------------------
# Precompute forecasts for all tickers
# ---------------------------------------------
for ticker in tickers:
    df_ticker = df_all[df_all["Ticker"] == ticker].copy().sort_values("Date").reset_index(drop=True)
    train = df_ticker.copy()
    test = df_ticker.copy()
    print(f"Computing forecasts for {ticker} ({len(df_ticker)} rows)...")
    arima = forecast_arima(train, test)
    rf = forecast_rf(train, test)
    prophet = forecast_prophet(train, test)
    combined = pd.DataFrame({
        "Date": df_ticker["Date"],
        "Actual": df_ticker["Close"],
        "ARIMA": arima.values,
        "RF": rf.values,
        "Prophet": prophet.values
    })
    output_file = os.path.join(output_dir, f"{ticker}_forecasts.csv")
    combined.to_csv(output_file, index=False)
    print(f"Saved forecasts for {ticker} → {output_file}")

print("✅ Finished precomputing forecasts for all tickers.")
