# scripts/compare_strategies.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils import download_data
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

# Compute project-root-relative results directory
BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TICKERS  = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "UNH", "NVDA"]
START    = "2018-01-01"
END      = "2024-12-31"
OUT_PLOT = RESULTS_DIR / "compare_frontier.png"

# 1) Download data & compute mu/S
prices = download_data(TICKERS, start=START, end=END)
mu     = expected_returns.mean_historical_return(prices)
S      = risk_models.sample_cov(prices)

# 2) Plot full efficient frontier
ef = EfficientFrontier(mu, S)
fig, ax = plt.subplots(figsize=(8, 5))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# 3) Load saved weight sets
mk = pd.read_csv(RESULTS_DIR / "markowitz_weights.csv",    index_col=0).squeeze("columns")
ml = pd.read_csv(RESULTS_DIR / "ml_optimizer_weights.csv", index_col=0).squeeze("columns")

# 4) Compute performance for each
def perf_from_weights(w):
    ef_t = EfficientFrontier(mu, S)
    ef_t.set_weights(w.to_dict())
    return ef_t.portfolio_performance()

mk_ret, mk_vol, mk_sr = perf_from_weights(mk)
ml_ret, ml_vol, ml_sr = perf_from_weights(ml)

# 5) Plot points
ax.scatter(mk_vol, mk_ret, marker="o", color="blue",
           label=f"Markowitz (SR={mk_sr:.2f})")
ax.scatter(ml_vol, ml_ret, marker="X", color="red",
           label=f"ML‐Opt (SR={ml_sr:.2f})")

ax.set_title("Efficient Frontier: Classical vs ML‐Optimized")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(OUT_PLOT)
print(f"👉 Saved comparison plot to {OUT_PLOT}")