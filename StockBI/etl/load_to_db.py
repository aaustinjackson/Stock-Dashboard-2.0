# etl/load_to_db.py
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ---------------------------
# Paths & DB connection
# ---------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data")

conn = psycopg2.connect(
    host="localhost",
    dbname="stockbi_db",
    user="postgres",
    password="Bengals",
    port="5432"
)
cursor = conn.cursor()

# ---------------------------
# Get all transformed CSVs sorted by filename
# ---------------------------
csv_files = sorted(
    [f for f in os.listdir(data_path) if f.startswith("transformed_") and f.endswith(".csv")]
)

for file_name in csv_files:
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)

    # Rename columns to match your SQL table
    df = df.rename(columns={
        "Adj Close": "adj_close",
        "Date": "date",
        "Ticker": "ticker",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "MA20": "ma20",
        "MA50": "ma50",
        "Daily Return": "daily_return",
        "Volatility20": "volatility20"
    })

    # Insert into PostgreSQL
    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ",".join(list(df.columns))
    query = f"INSERT INTO stock_data ({cols}) VALUES %s ON CONFLICT (date, ticker) DO NOTHING"
    execute_values(cursor, query, tuples)
    conn.commit()
    print(f"✅ Imported {file_name} into stock_data")

cursor.close()
conn.close()
print("🎉 All files imported successfully!")
