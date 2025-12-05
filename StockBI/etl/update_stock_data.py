import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------
# Configuration
# ---------------------------------------------
DATA_PATH = r"C:\Users\austi\PycharmProjects\StockBI\data\top10_stock_data_cleaned.csv"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOG", "NFLX", "AVGO", "JPM", "VRT"]
START_DATE = "2024-01-01"

# ---------------------------------------------
# Load existing data if available
# ---------------------------------------------
if os.path.exists(DATA_PATH):
    print("📂 Loading existing dataset...")
    df_existing = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df_existing["Ticker"] = df_existing["Ticker"].str.upper()
else:
    print("⚠️ No existing dataset found — starting fresh.")
    df_existing = pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])

# ---------------------------------------------
# Download updated data per ticker
# ---------------------------------------------
all_data = []

for ticker in TICKERS:
    print(f"\n📈 Processing {ticker}...")

    # Find last available date for this ticker
    last_date = df_existing[df_existing["Ticker"] == ticker]["Date"].max()
    if pd.notna(last_date):
        start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start = START_DATE

    end = datetime.now().strftime("%Y-%m-%d")

    if start >= end:
        print(f"✅ {ticker}: already up to date.")
        continue

    # Download from Yahoo Finance
    new_data = yf.download(ticker, start=start, end=end, progress=False)
    if new_data.empty:
        print(f"⚠️ No new data for {ticker}.")
        continue

    new_data.reset_index(inplace=True)
    new_data["Ticker"] = ticker
    all_data.append(new_data)

# ---------------------------------------------
# Merge, clean, and save
# ---------------------------------------------
if all_data:
    print("\n🧹 Merging and cleaning data...")
    new_df = pd.concat(all_data, ignore_index=True)
    new_df = new_df.rename(columns=str.capitalize)
    full_df = pd.concat([df_existing, new_df], ignore_index=True)

    # Sort, deduplicate, and save
    full_df = full_df.sort_values(["Ticker", "Date"]).drop_duplicates(subset=["Ticker", "Date"], keep="last")
    full_df.to_csv(DATA_PATH, index=False)
    print(f"\n💾 Updated data saved to:\n{DATA_PATH}")
    print(f"✅ Final rows: {len(full_df):,} | Tickers: {full_df['Ticker'].nunique()}")
else:
    print("\n✅ All tickers already up to date — no changes made.")
