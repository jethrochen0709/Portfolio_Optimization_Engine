# scripts/save_data_snapshot.py

import pandas as pd
import yfinance as yf
from pathlib import Path

# Define tickers and date range
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "UNH", "NVDA"]
start = "2018-01-01"
end = "2024-12-31"

# Download price data
df = yf.download(tickers, start=start, end=end, auto_adjust=True)

# Keep just adjusted close prices
if isinstance(df.columns, pd.MultiIndex):
    df = df["Close"]

# Ensure data directory exists
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Save to CSV
csv_path = DATA_DIR / "yfinance_prices_2018_2024.csv"
df.to_csv(csv_path)

print(f"✅ Saved historical price data to {csv_path}")