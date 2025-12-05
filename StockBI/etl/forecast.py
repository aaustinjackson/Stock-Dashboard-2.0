import pandas as pd
from prophet import Prophet
import psycopg2
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# 1. Connect to PostgreSQL
engine = create_engine("postgresql://postgres:Bengals@localhost:5432/stockbi_db")

# 2. Pick one ticker for testing
ticker = "AAPL"
query = f"SELECT date, close FROM stock_data WHERE ticker = '{ticker}' ORDER BY date;"
df = pd.read_sql(query, engine)

# 3. Format for Prophet (Prophet expects columns ds (date) and y (value))
df = df.rename(columns={"date": "ds", "close": "y"})

# 4. Build & fit the Prophet model
model = Prophet(daily_seasonality=True)
model.fit(df)

# 5. Make a future dataframe (e.g. forecast 90 days ahead)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# 6. Plot results
fig1 = model.plot(forecast)
plt.title(f"{ticker} Stock Price Forecast")
plt.show()

fig2 = model.plot_components(forecast)
plt.show()

# 7. Save forecast back to DB
forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_out["ticker"] = ticker
forecast_out.to_sql("forecasts", engine, if_exists="append", index=False)
print(f"✅ Forecast for {ticker} saved to DB")
