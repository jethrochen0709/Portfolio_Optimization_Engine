#!/usr/bin/env python3
"""
Feature builder + simple RF next-day return forecaster.

API:
- build_features(prices) -> pd.DataFrame
- predict_next_return(features, target_returns, min_history=60, return_model=False)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ───────────────────────────────────────────────────────── helpers
def _feat_cols(ticker: str) -> list[str]:
    return [
        f"{ticker}_mom5",
        f"{ticker}_mom21",
        f"{ticker}_vol5",
        f"{ticker}_vol21",
    ]


# ───────────────────────────────────────────────────────── features
def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-ticker momentum & vol features:
      - mom5, mom21: pct_change over 5/21 days
      - vol5, vol21: rolling std of daily returns over 5/21 days

    Parameters
    ----------
    prices : pd.DataFrame
        Wide dataframe of prices (columns=tickers).

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by date. Columns like 'AAPL_mom5', ...
    """
    prices = prices.sort_index()
    rets = prices.pct_change()

    feats = {}
    for c in prices.columns:
        feats[f"{c}_mom5"] = prices[c].pct_change(5)
        feats[f"{c}_mom21"] = prices[c].pct_change(21)
        feats[f"{c}_vol5"] = rets[c].rolling(5).std()
        feats[f"{c}_vol21"] = rets[c].rolling(21).std()

    X = pd.DataFrame(feats, index=prices.index)

    # Clean: finite only, drop incomplete rows
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return X


# ───────────────────────────────────────────────────────── model/predict
def predict_next_return(
    features: pd.DataFrame,
    target_returns: pd.Series,
    min_history: int = 60,
    return_model: bool = False,
):
    """
    Train a RandomForest to predict next-day return for one ticker and
    return the 1-step-ahead forecast for the *last* row in `features`.

    We infer the ticker from `target_returns.name` and pick that ticker's
    columns from `features`.

    Parameters
    ----------
    features : pd.DataFrame
        Output of build_features (all tickers included).
    target_returns : pd.Series
        Daily returns for ONE ticker (e.g., rets["AAPL"]).
    min_history : int
        Minimum number of (X,y) rows required to fit the model.
    return_model : bool
        If True, also return the fitted model.

    Returns
    -------
    float  (or (float, RandomForestRegressor) if return_model=True)
    """
    if not isinstance(target_returns, pd.Series):
        raise TypeError("target_returns must be a pandas Series for a single ticker")

    ticker = str(target_returns.name)
    if not ticker:
        raise ValueError("target_returns.name must be set to the ticker symbol")

    cols = _feat_cols(ticker)
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise ValueError(f"features missing expected columns for {ticker}: {missing}")

    # Align X and y; predict y_{t+1}
    X_all = features[cols]
    y_all = target_returns.reindex(X_all.index)

    # build supervised pairs: (X_t, y_{t+1})
    X_hist = X_all.iloc[:-1].copy()
    y_shift = y_all.shift(-1).iloc[:-1]

    # Drop any rows with NaNs in either X or y
    valid_idx = X_hist.dropna(how="any").index.intersection(y_shift.dropna().index)
    X_hist = X_hist.loc[valid_idx]
    y_hist = y_shift.loc[valid_idx]

    if len(y_hist) < min_history:
        # Not enough data: return a conservative neutral forecast
        pred = 0.0
        return (pred, None) if return_model else pred

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_hist, y_hist)

    # Predict the last available row (t = last index in features)
    X_last = X_all.iloc[[-1]].copy()
    X_last = X_last.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    pred = float(rf.predict(X_last)[0])

    return (pred, rf) if return_model else pred