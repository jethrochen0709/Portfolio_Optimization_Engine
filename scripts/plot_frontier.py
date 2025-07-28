# scripts/plot_frontier.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils import download_data
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

# Compute project-root-relative results directory
BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def plot_frontier(tickers, start="2018-01-01", end="2024-12-31"):
    # 1) Download price data
    df = download_data(tickers, start, end)

    # 2) Compute return expectations & covariance
    mu = expected_returns.mean_historical_return(df)
    S  = risk_models.sample_cov(df)

    # 3) Plot the Efficient Frontier
    ef_plot = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
    ax.set_title("Efficient Frontier with Asset Points")
    plt.tight_layout()
    fig_path = RESULTS_DIR / "efficient_frontier.png"
    fig.savefig(fig_path)
    print(f"👉 Saved plot to {fig_path}")

    # 4) Optimize for max Sharpe on a fresh instance
    ef_opt = EfficientFrontier(mu, S)
    ef_opt.max_sharpe()
    cleaned = ef_opt.clean_weights()

    # 5) Print & save weights and performance
    print("\nOptimal Weights:")
    for t, w in cleaned.items():
        if w > 0:
            print(f"  {t}: {w:.2%}")
    ef_opt.portfolio_performance(verbose=True)

    weights_path = RESULTS_DIR / "markowitz_weights.csv"
    pd.Series(cleaned).to_csv(weights_path)
    print(f"👉 Saved weights to {weights_path}")

if __name__ == "__main__":
    sample = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "UNH", "NVDA"]
    plot_frontier(sample)