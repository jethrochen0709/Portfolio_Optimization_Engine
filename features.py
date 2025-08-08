# features.py
from __future__ import annotations
import numpy as np
import pandas as pd

def build_features(prices: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Build per-ticker features from a price window.

    Accepts extra *args/**kwargs so older callers like
    `build_features(prices, tickers)` won't crash. Extras are ignored.

    For each ticker c, creates:
      c_mom5, c_mom21  : 5/21D simple momentum (%)
      c_vol5, c_vol21  : 5/21D realized vol (std of daily returns)
      c_ret1           : 1D lagged return
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame()

    px = prices.sort_index().astype(float)
    # daily returns with basic de-gunking
    rets = (
        px.pct_change()
          .replace([np.inf, -np.inf], np.nan)
          .clip(lower=-0.20, upper=0.20)
    )

    feats = {}
    for c in px.columns:
        feats[f"{c}_mom5"]  = px[c].pct_change(5)
        feats[f"{c}_mom21"] = px[c].pct_change(21)
        feats[f"{c}_vol5"]  = rets[c].rolling(5).std()
        feats[f"{c}_vol21"] = rets[c].rolling(21).std()
        feats[f"{c}_ret1"]  = rets[c].shift(1)

    F = pd.DataFrame(feats, index=px.index)
    F = F.replace([np.inf, -np.inf], np.nan).dropna()
    return F
