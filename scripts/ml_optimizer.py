# scripts/ml_optimizer.py

import pandas as pd
from pathlib import Path
from utils import download_data
from sklearn.ensemble import RandomForestRegressor
from pypfopt import EfficientFrontier, risk_models

# Compute project-root-relative results directory
BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def run_ml_optimizer(
    tickers=None,
    start="2018-01-01",
    end="2024-12-31"
):
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "UNH", "NVDA"]

    # 1) Download adjusted close prices & compute returns
    prices = download_data(tickers, start, end)
    rets = prices.pct_change().dropna()

    # 2) Forecast next-day return per ticker
    mu_pred = {}
    for ticker in tickers:
        r = rets[ticker]
        df_feat = pd.DataFrame({
            "r1":  r.shift(1),
            "r5":  r.rolling(5).mean().shift(1),
            "r10": r.rolling(10).mean().shift(1),
        }).dropna()
        if df_feat.empty:
            continue
        y = r.loc[df_feat.index]
        X = df_feat.values

        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mu_pred[ticker] = model.predict(X[-1].reshape(1, -1))[0]

    mu = pd.Series(mu_pred)
    S  = risk_models.sample_cov(prices)

    # 3) Optimize portfolio with predicted mu
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    cleaned = ef.clean_weights()

    # 4) Print & save ML‐optimized weights
    print("ML‐Optimized Weights:")
    for t, w in cleaned.items():
        if w > 0:
            print(f"  {t}: {w:.2%}")
    ef.portfolio_performance(verbose=True)

    out_path = RESULTS_DIR / "ml_optimizer_weights.csv"
    pd.Series(cleaned).to_csv(out_path)
    print(f"👉 Saved ML weights to {out_path}")

if __name__ == "__main__":
    run_ml_optimizer()