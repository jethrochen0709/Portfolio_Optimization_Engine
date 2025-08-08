#!/usr/bin/env python3
"""
main.py walk-forward back-test comparing
1) Markowitz (μ = hist. mean)   vs.
2) ML forecast   (μ = RandomForest one-day-ahead)

Uses period-based downloads, tighter return cleaning, and quiet LW fallback.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from pypfopt import risk_models, EfficientFrontier
from evaluate  import sharpe_ratio, cagr, max_drawdown
from visuals   import plot_equity

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# ── config
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "UNH", "NVDA"]
PERIOD_DAYS = 3 * 252
RISK_FREE = 0.02

# ── helpers
def fetch_prices(tickers, periods: int):
    # Explicit auto_adjust=True to avoid the FutureWarning and get adjusted closes
    data = yf.download(tickers, period=f"{periods}d", progress=False, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        df = data["Close"] if "Close" in data.columns.get_level_values(0) else data.iloc[:, 0]
    else:
        df = data

    df = df.sort_index().ffill().dropna(how="all", axis=0)
    return df

def clean_returns(ret_df):
    """Remove inf / overflow, clip +/-10 %, drop rows/cols with NaNs."""
    ret_df = ret_df.replace([np.inf, -np.inf], np.nan)
    ret_df = ret_df.clip(lower=-0.10, upper=0.10)
    ret_df = ret_df.dropna(axis="columns", how="all")
    ret_df = ret_df.dropna(axis="index",   how="any")
    return ret_df

def train_rf(X, y):
    rf = RandomForestRegressor(
        n_estimators=120, max_depth=5, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    return rf

def ml_expected_returns(price_window: pd.DataFrame) -> pd.Series:
    """
    One-day-ahead RF forecast for each asset; returns pd.Series of DAILY μ.
    (Fixed) Build features column-wise per asset instead of the broken concat trick.
    """
    rets = price_window.pct_change().dropna()

    # Build a single features DataFrame with all assets’ features as columns
    feat_blocks = []
    for c in price_window.columns:
        block = pd.DataFrame({
            f"{c}_mom5" : price_window[c].pct_change(5),
            f"{c}_mom21": price_window[c].pct_change(21),
            f"{c}_vol5" : rets[c].rolling(5).std(),
            f"{c}_vol21": rets[c].rolling(21).std(),
        })
        feat_blocks.append(block)

    feat = pd.concat(feat_blocks, axis=1)
    feat = feat.replace([np.inf, -np.inf], np.nan).ffill().dropna(how="any")

    preds = {}
    for c in price_window.columns:
        cols = [f"{c}_mom5", f"{c}_mom21", f"{c}_vol5", f"{c}_vol21"]

        # Align X and y on the same index, predict t+1 return
        X = feat[cols].iloc[:-1]
        y = rets[c].shift(-1).reindex(X.index).dropna()
        X = X.loc[y.index]

        if len(y) < 50:  # not enough history -> neutral
            preds[c] = 0.0
            continue

        model = train_rf(X, y)
        X_last = feat[cols].iloc[[-1]].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        preds[c] = float(model.predict(X_last)[0])

    return pd.Series(preds)

def robust_cov(rets_hist):
    """Ledoit-Wolf shrinkage with safe fall-back (warnings silenced globally)."""
    try:
        return risk_models.CovarianceShrinkage(rets_hist).ledoit_wolf()
    except Exception as e:
        print(f"⚠️  shrinkage failed ({e}); using sample covariance.")
        return rets_hist.cov()

def robust_optimize(mu, cov, long_only=False):
    ef = EfficientFrontier(mu * 252, cov, solver="ECOS")  # μ annualised
    try:
        if (mu <= RISK_FREE / 252).all():
            ef.min_volatility()
        else:
            ef.max_sharpe(risk_free_rate=RISK_FREE)
    except Exception as e:
        print(f"⚠️  optimiser failed ({e}); equal-weighting.")
        w = np.repeat(1/len(mu), len(mu))
        return dict(zip(mu.index, w))
    return ef.clean_weights()

# ── walk-forward
def walk_forward_backtest(prices, train_days=252):
    if len(prices) <= train_days + 1:
        new_train = min(max(60, len(prices) - 2), train_days)
        print(f"⚠️  Not enough history for train_days={train_days}. Using {new_train} instead.")
        train_days = new_train

    returns = prices.pct_change().dropna()
    eq_mk, eq_ml = [1.0], [1.0]

    for end_ix in range(train_days, len(prices)-1):
        window_prices = prices.iloc[end_ix-train_days:end_ix]
        tomorrow      = prices.index[end_ix+1]

        # MARKOWITZ μ/Σ
        rets_hist = clean_returns(window_prices.pct_change().dropna())
        mu_mk     = rets_hist.mean()
        S         = robust_cov(rets_hist)
        w_mk      = robust_optimize(mu_mk, S)

        # ML μ/Σ
        mu_ml = ml_expected_returns(window_prices)
        w_ml  = robust_optimize(mu_ml, S)

        # realise next-day return
        r_next = returns.loc[tomorrow]
        eq_mk.append(eq_mk[-1] * (1 + np.dot(r_next.fillna(0),  pd.Series(w_mk))))
        eq_ml.append(eq_ml[-1] * (1 + np.dot(r_next.fillna(0),  pd.Series(w_ml))))

    idx = prices.index[train_days:]
    return pd.Series(eq_mk, index=idx[:len(eq_mk)]), pd.Series(eq_ml, index=idx[:len(eq_ml)])

# ── main
def main():
    print("Downloading price data…")
    prices = fetch_prices(TICKERS, periods=PERIOD_DAYS)

    print("Running walk-forward backtest…")
    eq_mk, eq_ml = walk_forward_backtest(prices)

    start_d = eq_mk.index.min().date() if len(eq_mk) else prices.index.min().date()
    end_d   = eq_mk.index.max().date() if len(eq_mk) else prices.index.max().date()
    print(f"\nResults {start_d} → {end_d}")
    print("Markowitz  : SR={:.2f} | CAGR={:.2%} | MDD={:.2%}".format(
        sharpe_ratio(eq_mk), cagr(eq_mk), max_drawdown(eq_mk)))
    print("ML forecast: SR={:.2f} | CAGR={:.2%} | MDD={:.2%}".format(
        sharpe_ratio(eq_ml), cagr(eq_ml), max_drawdown(eq_ml)))

    os.makedirs("results", exist_ok=True)
    plot_equity({"Markowitz": eq_mk, "ML": eq_ml}, "results/equity_comparison.png")
    print("Chart saved to results/equity_comparison.png")
    print("✅ All done!")

if __name__ == "__main__":
    main()
