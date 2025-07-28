# scripts/markowitz.py

import pandas as pd
from pathlib import Path
from utils import download_data
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Compute project-root-relative results directory
BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def run_markowitz():
    # 1) Download price data
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "UNH", "NVDA"]
    df = download_data(tickers, start="2018-01-01", end="2024-12-31")

    # 2) Compute return expectations & covariance
    mu = expected_returns.mean_historical_return(df)
    S  = risk_models.sample_cov(df)

    # 3) Optimize for max Sharpe ratio
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    cleaned = ef.clean_weights()

    # 4) Print results
    print("Optimal Weights:")
    for ticker, w in cleaned.items():
        if w > 0:
            print(f"  {ticker}: {w:.2%}")
    ef.portfolio_performance(verbose=True)

    # 5) Save weights
    out_path = RESULTS_DIR / "markowitz_weights.csv"
    pd.Series(cleaned).to_csv(out_path)
    print(f"👉 Saved weights to {out_path}")

if __name__ == "__main__":
    run_markowitz()