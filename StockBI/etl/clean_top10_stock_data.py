import pandas as pd
import os

# Adjust path to your actual file
data_path = r"C:\Users\austi\PycharmProjects\StockBI\data\top10_stock_data.csv"
cleaned_path = r"C:\Users\austi\PycharmProjects\StockBI\data\top10_stock_data_cleaned.csv"

print("🔍 Loading raw stock data...")
df = pd.read_csv(data_path, parse_dates=["Date"])
print(f"Loaded {len(df):,} rows")

# Basic cleaning
df = df.dropna(subset=["Date", "Ticker", "Close"])
df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

# Sort by ticker/date
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# Remove duplicates (keep most recent data for each Ticker-Date)
df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")

# Optional: detect missing date gaps
gaps = []
for t, sub in df.groupby("Ticker"):
	sub = sub.sort_values("Date")
	diffs = sub["Date"].diff().dt.days
	missing_days = (diffs > 3).sum()
	if missing_days > 0:
		gaps.append((t, missing_days))
if gaps:
	print("\n⚠️ Potential gaps found:")
	for t, g in gaps:
		print(f"  {t}: {g} missing intervals")
else:
	print("\n✅ No large gaps detected")

# Save cleaned dataset
df.to_csv(cleaned_path, index=False)
print(f"\n💾 Cleaned data saved to:\n{cleaned_path}")
print(f"✅ Final dataset: {len(df):,} rows across {df['Ticker'].nunique()} tickers")
