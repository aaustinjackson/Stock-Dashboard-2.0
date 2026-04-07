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
# ARIMA
# ---------------------------------------------
def forecast_arima(train_df, test_df, window=5):
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
        history.append(pred)  # no leakage

    return pd.Series(preds, index=test_df["Date"])


# ---------------------------------------------
# Random Forest (FIXED)
# ---------------------------------------------
def forecast_rf(train_df, test_df, lags=10):
    rf_data = train_df.copy().sort_values("Date")

    # Lag features
    for lag in range(1, lags + 1):
        rf_data[f"lag_{lag}"] = rf_data["Close"].shift(lag)

    # Additional features (IMPORTANT)
    rf_data["returns"] = rf_data["Close"].pct_change()
    rf_data["rolling_mean_5"] = rf_data["Close"].rolling(5).mean()
    rf_data["rolling_std_5"] = rf_data["Close"].rolling(5).std()
    rf_data["day_of_week"] = rf_data["Date"].dt.dayofweek

    rf_data = rf_data.dropna()

    if rf_data.empty:
        val = float(train_df["Close"].iloc[-1])
        return pd.Series([val] * len(test_df), index=test_df["Date"])

    feature_cols = (
        [f"lag_{i}" for i in range(1, lags + 1)] +
        ["returns", "rolling_mean_5", "rolling_std_5", "day_of_week"]
    )

    X_train = rf_data[feature_cols].values
    y_train = rf_data["Close"].values

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Rolling predictions (NO LEAKAGE)
    all_closes = list(train_df.sort_values("Date")["Close"].astype(float).values)

    preds = []

    for i in range(len(test_df)):
        if len(all_closes) < lags:
            history_window = [all_closes[0]] * (lags - len(all_closes)) + all_closes
        else:
            history_window = all_closes[-lags:]

        # Build feature row
        temp_series = pd.Series(all_closes)

        returns = temp_series.pct_change().iloc[-1] if len(temp_series) > 1 else 0
        rolling_mean_5 = temp_series.rolling(5).mean().iloc[-1]
        rolling_std_5 = temp_series.rolling(5).std().iloc[-1]
        day_of_week = test_df["Date"].iloc[i].dayofweek

        feature_row = np.array(
            history_window + [returns, rolling_mean_5, rolling_std_5, day_of_week]
        ).reshape(1, -1)

        pred = model.predict(feature_row)[0]
        preds.append(pred)

        # IMPORTANT: only use prediction (no actual leakage)
        all_closes.append(pred)

    return pd.Series(preds, index=test_df["Date"])


# ---------------------------------------------
# Prophet
# ---------------------------------------------
def forecast_prophet(train_df, test_df):
    if len(train_df) < 2:
        val = train_df["Close"].iloc[-1] if len(train_df) > 0 else 0
        return pd.Series([val] * len(test_df), index=test_df["Date"])

    history = train_df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05
    )

    model.fit(history)

    future = model.make_future_dataframe(periods=len(test_df), freq='D')
    forecast = model.predict(future)

    forecast_series = forecast.set_index("ds")["yhat"]
    aligned = forecast_series.reindex(pd.to_datetime(test_df["Date"]), method="nearest")
    aligned.index = pd.to_datetime(test_df["Date"])

    return aligned


# ---------------------------------------------
# Run forecasts
# ---------------------------------------------
for ticker in tickers:
    df_ticker = (
        df_all[df_all["Ticker"] == ticker]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # ✅ Proper split
    split = int(len(df_ticker) * 0.8)
    train = df_ticker.iloc[:split]
    test = df_ticker.iloc[split:]

    print(f"Computing forecasts for {ticker}...")

    arima = forecast_arima(train, test)
    rf = forecast_rf(train, test)
    prophet = forecast_prophet(train, test)

    combined = pd.DataFrame({
        "Date": test["Date"],
        "Actual": test["Close"].values,
        "ARIMA": arima.values,
        "RF": rf.values,
        "Prophet": prophet.values
    })

    output_file = os.path.join(output_dir, f"{ticker}_forecasts.csv")

    if ticker == "GOOG":
        output_file = os.path.join(output_dir, "GOOGL_forecasts.csv")

    combined.to_csv(output_file, index=False)

    print(f"Saved → {output_file}")

print("✅ Finished all forecasts.")
