import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "top10_stock_data_cleaned.csv")
plots_path = os.path.join(project_root, "plots")
os.makedirs(plots_path, exist_ok=True)

# ---------------------------
# Load and transform data
# ---------------------------
def load_and_transform(ticker):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data file not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df[df["Ticker"] == ticker].copy()

    if df.empty:
        raise ValueError(f"❌ No data found for ticker {ticker}")

    numeric_cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Technical indicators
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Daily Return"] = df["Close"].pct_change()
    df["Volatility20"] = df["Daily Return"].rolling(20).std()

    return df

# ---------------------------
# Plot functions
# ---------------------------
def plot_price_with_ma(df, ticker):
    plt.figure(figsize=(10,6))
    plt.plot(df["Date"], df["Close"], label="Close Price", alpha=0.8)
    if "MA20" in df.columns: plt.plot(df["Date"], df["MA20"], linestyle="--", label="MA20")
    if "MA50" in df.columns: plt.plot(df["Date"], df["MA50"], linestyle="--", label="MA50")
    plt.title(f"{ticker} - Price with Moving Averages")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"{ticker}_ma.png")); plt.close()

def plot_returns_distribution(df, ticker):
    if "Daily Return" not in df.columns: return
    plt.figure(figsize=(8,5))
    df["Daily Return"].hist(bins=50, alpha=0.7)
    plt.title(f"{ticker} - Daily Returns Distribution")
    plt.xlabel("Daily Return"); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"{ticker}_returns.png")); plt.close()

def plot_volatility(df, ticker):
    if "Volatility20" not in df.columns: return
    plt.figure(figsize=(10,6))
    plt.plot(df["Date"], df["Volatility20"], label="20-day Volatility", color="orange")
    plt.title(f"{ticker} - Rolling Volatility")
    plt.xlabel("Date"); plt.ylabel("Volatility")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"{ticker}_volatility.png")); plt.close()

def plot_cumulative_returns(df, ticker):
    if "Daily Return" not in df.columns: return
    cum_return = (1 + df["Daily Return"]).cumprod()
    plt.figure(figsize=(10,6))
    plt.plot(df["Date"], cum_return, label="Cumulative Return", color="green")
    plt.title(f"{ticker} - Cumulative Returns since {df['Date'].min().date()}")
    plt.xlabel("Date"); plt.ylabel("Cumulative Return")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"{ticker}_cumulative.png")); plt.close()

# ---------------------------
# Run for all tickers (including VRT)
# ---------------------------
if __name__ == "__main__":
    top10_tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "UNH", "TSLA", "JPM", "VRT"]

    for ticker in top10_tickers:
        try:
            df = load_and_transform(ticker)
            df.to_csv(os.path.join(project_root, "data", f"transformed_{ticker}.csv"), index=False)
            print(f"✅ Transformed and saved {ticker}, {len(df)} rows")

            plot_price_with_ma(df, ticker)
            plot_returns_distribution(df, ticker)
            plot_volatility(df, ticker)
            plot_cumulative_returns(df, ticker)
            print(f"📊 Plots saved for {ticker}\n")

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}\n")
