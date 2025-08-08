# backtester.py
"""
Lightweight backtesting utilities.

- simple_backtest: buy-and-hold (static weights).
  Accepts prices as pd.DataFrame OR numpy array/list.
  If `weights` is None, uses equal-weight across columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Mapping, Sequence, Union

Weights = Union[Mapping[str, float], Sequence[float], np.ndarray, pd.Series, None]


def _to_price_df(prices) -> pd.DataFrame:
    """Coerce input prices into a DataFrame with sensible index/columns."""
    if isinstance(prices, pd.DataFrame):
        return prices
    # assume array-like
    arr = np.asarray(prices, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    idx = pd.RangeIndex(len(arr))
    cols = [f"A{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=idx, columns=cols)


def _as_weight_series(cols: pd.Index, weights: Weights) -> pd.Series:
    """Convert various weight inputs into a Series aligned to columns."""
    n = len(cols)
    if weights is None:
        return pd.Series(np.full(n, 1.0 / n), index=cols, dtype=float)

    if isinstance(weights, pd.Series):
        w = weights.reindex(cols).fillna(0.0).astype(float)
    elif isinstance(weights, dict):
        w = pd.Series({c: weights.get(c, 0.0) for c in cols}, index=cols, dtype=float)
    else:
        # assume sequence-like in the same order as columns
        w = pd.Series(np.asarray(weights, dtype=float), index=cols)

    s = w.sum()
    if np.isfinite(s) and s != 0:
        w = w / s
    return w


def simple_backtest(prices, weights: Weights = None) -> np.ndarray:
    """
    Buy-and-hold backtest with static weights.

    Parameters
    ----------
    prices : DataFrame | array-like
        Wide prices (Date x Tickers). If array-like, it's coerced to DataFrame.
    weights : mapping/sequence/Series/None
        Portfolio weights matching the columns. If None, equal-weight.

    Returns
    -------
    np.ndarray
        Equity curve starting at 1.0 (length == len(prices)).
    """
    if prices is None:
        return np.array([1.0], dtype=float)

    px = _to_price_df(prices)

    # Clean/sort
    px = (
        px.astype(float)
        .sort_index()
        .ffill()
        .dropna(how="any", axis=1)  # drop tickers with missing data
    )
    if px.shape[1] == 0 or px.shape[0] == 0:
        return np.array([1.0], dtype=float)

    # Align weights (defaults to equal-weight if None)
    w = _as_weight_series(px.columns, weights)

    # Daily returns
    rets = px.pct_change().fillna(0.0)

    # Portfolio daily return
    port_ret = rets.dot(w.values)

    # Safety: avoid <-100% or absurd spikes from dirty inputs
    port_ret = port_ret.clip(lower=-0.99, upper=0.99)

    # Equity curve
    equity = (1.0 + port_ret).cumprod().values
    return equity

