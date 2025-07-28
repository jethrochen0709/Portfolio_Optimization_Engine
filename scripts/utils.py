# scripts/utils.py

import yfinance as yf
import pandas as pd

def download_data(tickers, start="2018-01-01", end="2024-12-31"):
    """
    Download adjusted close price data for the given tickers.
    Returns a DataFrame of shape (dates, tickers) with no missing columns.
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )
    # Extract the 'Close' column for all tickers
    if isinstance(df.columns, pd.MultiIndex):
        price_df = df["Close"]
    else:
        price_df = df["Close"].to_frame()

    # Drop any ticker columns with missing data
    price_df = price_df.dropna(axis=1, how="any")
    return price_df