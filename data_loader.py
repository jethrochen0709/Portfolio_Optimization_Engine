# data_loader.py
import pandas as pd
import yfinance as yf

def load_price_data(tickers, periods=252, start=None, end=None, price_col="Close"):
    """
    Load price data into a (Date x Tickers) DataFrame of prices.
    - If start & end are given, use that range; otherwise use `periods` trading days.
    - Tries `price_col` first, then falls back to 'Adj Close' or 'Close'.
    """
    df_list = []
    for t in tickers:
        if start and end:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        else:
            df = yf.download(t, period=f"{periods}d", progress=False, auto_adjust=True)

        # pick a price series robustly
        if price_col in df.columns:
            price = df[price_col]
        elif "Adj Close" in df.columns:
            price = df["Adj Close"]
        elif "Close" in df.columns:
            price = df["Close"]
        else:
            # nothing usable for this ticker; skip it
            continue

        s = price.astype(float).rename_axis("Date")  # keep index name
        s.name = t                                   # <-- set the series name properly
        df_list.append(s)

    if not df_list:
        return pd.DataFrame()

    out = (
        pd.concat(df_list, axis=1)
        .sort_index()
        .ffill()               # forward-fill holidays/missing days
        .dropna(how="all")     # drop rows that are entirely NaN
    )
    return out
