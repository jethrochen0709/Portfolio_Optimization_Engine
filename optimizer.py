from collections import OrderedDict
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier

def optimize_mv(mu, cov, long_only=True, leverage=1.0, risk_free_rate=0.0):
    """
    Convenience wrapper for classic MV max-sharpe.
    - mu: dict/Series of DAILY expected returns
    - cov: DataFrame covariance (DAILY)
    """
    if not isinstance(mu, pd.Series):
        mu = pd.Series(mu)
    bounds = (0.0, leverage) if long_only else (-leverage, leverage)
    ef = EfficientFrontier(mu * 252.0, cov, weight_bounds=bounds, solver="ECOS")

    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        w = ef.clean_weights()
    except Exception:
        n = len(mu)
        w = {k: 1.0 / n for k in mu.index}

    # round for readability like your earlier output
    w = {k: float(round(v, 5)) for k, v in w.items() if abs(v) > 1e-6}
    return OrderedDict(sorted(w.items()))
