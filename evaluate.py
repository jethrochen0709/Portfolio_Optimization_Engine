import numpy as np
import pandas as pd

def sharpe_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute annualized Sharpe ratio from an equity curve.
    """
    returns = equity.pct_change().dropna()
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()

def max_drawdown(equity: pd.Series) -> float:
    """
    Calculate maximum drawdown of an equity curve.
    """
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return drawdown.min()

def cagr(equity: pd.Series) -> float:
    """
    Compute compound annual growth rate (CAGR) of an equity curve.
    """
    start_date = equity.index[0]
    end_date = equity.index[-1]
    years = (end_date - start_date).days / 365.25
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1