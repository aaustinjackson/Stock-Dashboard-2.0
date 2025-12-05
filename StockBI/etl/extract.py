import yfinance as yf
import pandas as pd
import os
import time
import random
from datetime import datetime

# ---------------------------
# Paths
# ---------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "top10_stock_data.csv")
os.makedirs(os.path.join(project_root, "data"), exist_ok=True)

# ---------------------------
# Top 10 tickers
# ---------------------------
top10_tickers = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
    "META", "BRK-B", "UNH", "TSLA", "JPM" , "VRT"
]

# ---------------------------
# Get last saved date per ticker
# ---------------------------
def get_last_dates():
    if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
        return {}
    df = pd.read_csv(data_path, parse_dates=["Date"])
    return df.groupby("Ticker")["Date"].max().to_dict()

# ---------------------------
# Safe single-ticker download
# ---------------------------
def safe_download(ticker, start, end, max_retries=5, base_delay=30):
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, threads=False)
            if df.empty:
                raise ValueError("No data returned")
            df.reset_index(inplace=True)
            df["Ticker"] = ticker
            return df
        except Exception as e:
            wait = base_delay * attempt + random.randint(0, 10)
            print(f"❌ Attempt {attempt}/{max_retries} failed for {ticker}: {e}. Waiting {wait} sec...")
            time.sleep(wait)
    print(f"⚠️ {ticker} failed after {max_retries} retries. Skipping.")
    return None

# ---------------------------
# Fetch and save one ticker at a time
# ---------------------------
def fetch_and_save(tickers):
    today = datetime.today().strftime("%Y-%m-%d")
    last_dates = get_last_dates()
    global_start = "2024-01-01"

    for ticker in tickers:
        if ticker in last_dates:
            start_date = (last_dates[ticker] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start_date = global_start

        print(f"📈 Fetching {ticker} from {start_date} to {today}...")

        df = safe_download(ticker, start=start_date, end=today)
        if df is None or df.empty:
            print(f"⚠️ Skipped {ticker} (no new rows)")
            continue

        write_header = not os.path.exists(data_path) or os.path.getsize(data_path) == 0
        df.to_csv(data_path, mode="a", header=write_header, index=False)
        print(f"✅ Saved {ticker} ({len(df)} rows).")

        # Sleep between tickers to respect rate limits
        wait_time = 30 + random.randint(0, 20)
        print(f"⏱ Waiting {wait_time} sec before next ticker...")
        time.sleep(wait_time)

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting incremental download of tickers...")
    fetch_and_save(top10_tickers)
    print("🎉 Finished updating all tickers!")
