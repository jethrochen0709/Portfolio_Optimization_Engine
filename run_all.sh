#!/usr/bin/env bash
set -e

echo
echo "⏳ 1) Testing data_loader..."
python - <<'PY'
import warnings
warnings.filterwarnings("ignore")
from data_loader import load_price_data

prices = load_price_data(["AAPL","MSFT"], periods=252)
print(f"  → Loaded: {prices.shape}")
PY

echo "⏳ 2) Testing optimizer..."
python - <<'PY'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from optimizer import optimize_mv

# Tiny synthetic example so this test is stable
prices = pd.DataFrame({
    "AAPL":[100,101,102,103,104,105,106,107],
    "MSFT":[100,100,101,101,102,103,103,104],
}, dtype=float)
rets = prices.pct_change().dropna()
mu = rets.mean()
S  = rets.cov()

w = optimize_mv(mu, S, long_only=True, leverage=1.0)
# Normalize to 5 dp for pretty output
w = {k: round(float(v),5) for k,v in w.items()}
from collections import OrderedDict
print("  → Weights:", OrderedDict(sorted(w.items())))
PY

echo "⏳ 3) Testing backtester..."
python - <<'PY'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from backtester import simple_backtest

# Two-day toy series so we can exercise the backtester quickly
prices = pd.DataFrame({
    "AAPL":[100, 101, 102],
    "MSFT":[100,  99, 100],
}, dtype=float)

weights = {"AAPL":0.6, "MSFT":0.4}
eq = simple_backtest(prices, weights)
print("  → Equity:", list(eq))
PY

echo "⏳ 4) Smoke test: features & models…"
python - <<'PY'
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from data_loader import load_price_data
from models import build_features, predict_next_return

# Grab a small window for a quick model sanity check
prices = load_price_data(["AAPL","MSFT"], periods=252)
rets   = prices.pct_change().dropna()

X = build_features(prices)

# Ask for both the prediction and the model to ensure the API works
pred, _ = predict_next_return(X, rets["AAPL"], min_history=50, return_model=True)

print(f"  → AAPL next-day RF pred: {pred:.6f}")
PY

echo "✅ All smoke tests passed."
