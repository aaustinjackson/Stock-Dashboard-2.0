import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from datetime import timedelta
import numpy as np

# ---------------------------------------------
# Paths
# ---------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "top10_stock_data.csv")
precomputed_dir = os.path.join(project_root, "data", "precomputed_forecasts")

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st.title("Stock Forecast Dashboard")
st.write("Compare ARIMA, Random Forest, and Prophet model forecasts.")

# ---------------------------------------------
# Load available tickers
# ---------------------------------------------
df_all = pd.read_csv(data_path, parse_dates=["Date"])
tickers = df_all["Ticker"].dropna().astype(str).unique()
tickers = sorted(tickers)
ticker = st.selectbox("Select a stock:", tickers)

# ---------------------------------------------
# Load precomputed forecasts
# ---------------------------------------------
forecast_file = os.path.join(precomputed_dir, f"{ticker}_forecasts.csv")
if not os.path.exists(forecast_file):
    st.error(f"No precomputed forecasts found for {ticker}. Run preprocessing first.")
    st.stop()

df = pd.read_csv(forecast_file, parse_dates=["Date"])
for col in ["Actual", "ARIMA", "RF", "Prophet"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------
# Remove initial NA and warm-up spikes
# ---------------------------------------------
df = df[df["Actual"].notna()].copy()
df.reset_index(drop=True, inplace=True)
if len(df) > 1:
    df = df.iloc[1:].copy()  # remove first warm-up row

# Drop rows where all forecasts are missing
df = df.dropna(subset=["ARIMA", "RF", "Prophet"], how="all")

# ---------------------------------------------
# Smooth forecasts and remove spikes robustly
# ---------------------------------------------
forecast_cols = ["ARIMA", "RF", "Prophet"]

# 1️⃣ Rolling median smoothing
for col in forecast_cols:
    df[col] = df[col].rolling(window=3, center=True, min_periods=1).median()

# 2️⃣ Cap extreme jumps relative to previous value
def cap_jumps(series, max_pct=0.15):
    """Cap percentage changes between consecutive points."""
    series = series.copy().reset_index(drop=True)  # ensure 0..N index
    for i in range(1, len(series)):
        prev = series.iloc[i-1]
        cur = series.iloc[i]
        if prev == 0:
            continue
        change = (cur - prev) / abs(prev)
        if abs(change) > max_pct:
            series.iloc[i] = prev * (1 + np.sign(change) * max_pct)
    return series


# Apply capping
df["RF"] = cap_jumps(df["RF"], max_pct=0.15)
df["ARIMA"] = cap_jumps(df["ARIMA"], max_pct=0.3)
df["Prophet"] = cap_jumps(df["Prophet"], max_pct=0.3)

# ---------------------------------------------
# Top controls: Date Range Selector + Next-Day Forecasts
# ---------------------------------------------
st.subheader("Controls & Forecasts")
col1, col2 = st.columns([2, 1])

# --- Column 1: Date Range Selector ---
with col1:
    range_option = st.radio(
        "Choose period:",
        ["1 Week", "1 Month", "1 Year", "All Data"],
        index=1,
        horizontal=True
    )

    min_date = df["Date"].min()
    max_date = df["Date"].max()

    if range_option == "1 Week":
        start_date = max_date - pd.Timedelta(days=7)
    elif range_option == "1 Month":
        start_date = max_date - pd.Timedelta(days=30)
    elif range_option == "1 Year":
        start_date = max_date - pd.Timedelta(days=365)
    else:
        start_date = min_date

    start_date = max(start_date, min_date)

    df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= max_date)].copy()
    if df_filtered.empty:
        st.warning("No data available in the selected date range.")
        st.stop()

    st.write(f"Selected date range: {df_filtered['Date'].min().date()} → {df_filtered['Date'].max().date()}")

# --- Column 2: Next-Day Forecasts ---
with col2:
    next_date = df_filtered["Date"].max() + pd.Timedelta(days=1)

    def next_day_forecast(df):
        next_arima = df["ARIMA"].iloc[-1]
        next_rf = df["RF"].iloc[-1]
        next_prophet = df["Prophet"].iloc[-1]
        return next_arima, next_rf, next_prophet

    next_arima, next_rf, next_prophet = next_day_forecast(df_filtered)

    def fmt(val):
        return f"{val:.2f}" if pd.notna(val) else "N/A"

    st.write(f"**Predictions for {next_date.strftime('%Y-%m-%d')}:**")
    st.write(f"🔴 ARIMA: {fmt(next_arima)}")
    st.write(f"🟢 Random Forest: {fmt(next_rf)}")
    st.write(f"🔵 Prophet: {fmt(next_prophet)}")

# ---------------------------------------------
# Compute forecast errors
# ---------------------------------------------
arima_error = df_filtered["Actual"] - df_filtered["ARIMA"]
rf_error = df_filtered["Actual"] - df_filtered["RF"]
prophet_error = df_filtered["Actual"] - df_filtered["Prophet"]

# ---------------------------------------------
# Plotting
# ---------------------------------------------
show_errors = st.checkbox("Show forecast errors instead of predicted prices", value=False)
fig, ax = plt.subplots(figsize=(12, 6))

if not show_errors:
    ax.scatter(df_filtered["Date"], df_filtered["Actual"], label="Actual Prices", s=30, zorder=3)
    ax.plot(df_filtered["Date"], df_filtered["ARIMA"], label="ARIMA Forecast", color="red", linewidth=2)
    ax.plot(df_filtered["Date"], df_filtered["RF"], label="Random Forest Forecast", color="green", linewidth=2)
    ax.plot(df_filtered["Date"], df_filtered["Prophet"], label="Prophet Forecast", color="blue", linewidth=2)
    ax.set_ylabel("Close Price")
    ax.set_title(f"{ticker} Close Price Forecast")
else:
    ax.plot(df_filtered["Date"], arima_error, label="ARIMA Error", color="red", linewidth=2)
    ax.plot(df_filtered["Date"], rf_error, label="RF Error", color="green", linewidth=2)
    ax.plot(df_filtered["Date"], prophet_error, label="Prophet Error", color="blue", linewidth=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Error")
    ax.autoscale(enable=True, axis='y')

ax.set_xlabel("Date")
ax.legend()
ax.grid(True)
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
fig.autofmt_xdate(rotation=25)

st.pyplot(fig, use_container_width=True)

