"""
Performance metrics: Total return, CAGR, Volatility, Sharpe, Sortino, Max drawdown.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .utils import TRADING_DAYS_PER_YEAR


def total_return(equity_curve: pd.Series) -> float:
    """Total return (decimal) from first to last value."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] == 0:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def cagr(
    equity_curve: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compound annual growth rate (decimal)."""
    tr = total_return(equity_curve)
    n = len(equity_curve) - 1
    if n <= 0 or tr <= -1:
        return 0.0
    years = n / trading_days_per_year
    if years <= 0:
        return 0.0
    return float((1 + tr) ** (1 / years) - 1)


def annualized_volatility(
    returns: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized volatility (decimal)."""
    if returns.empty or len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(trading_days_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio (excess return / volatility)."""
    if returns.empty or len(returns) < 2:
        return 0.0
    rf_daily = risk_free_rate / trading_days_per_year
    excess = returns - rf_daily
    vol = returns.std() * np.sqrt(trading_days_per_year)
    if vol == 0:
        return 0.0
    return float(excess.mean() * trading_days_per_year / vol)


def downside_deviation(
    returns: pd.Series,
    mar: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Downside deviation (annualized): std of returns below MAR."""
    if returns.empty:
        return 0.0
    below = returns[returns < mar]
    if len(below) == 0:
        return 0.0
    return float(below.std() * np.sqrt(trading_days_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    mar: Optional[float] = None,
) -> float:
    """Sortino ratio: (annualized excess return) / downside deviation."""
    if returns.empty or len(returns) < 2:
        return 0.0
    mar_ann = mar if mar is not None else risk_free_rate
    rf_daily = risk_free_rate / trading_days_per_year
    excess_ann = (returns.mean() - rf_daily) * trading_days_per_year
    dd = downside_deviation(returns, mar=mar_ann / trading_days_per_year, trading_days_per_year=trading_days_per_year)
    if dd == 0:
        return 0.0
    return float(excess_ann / dd)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown (decimal, e.g. -0.15 for -15%)."""
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    return float(drawdown.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Time series of drawdown (decimal)."""
    if equity_curve.empty:
        return pd.Series(dtype=float)
    cummax = equity_curve.cummax()
    return (equity_curve - cummax) / cummax


def var_historical(returns: pd.Series, confidence: float = 95.0) -> float:
    """
    Historical Value-at-Risk (1-day): loss level (positive) that is exceeded
    (1 - confidence/100) of the time. Uses quantile of daily returns.
    """
    if returns.empty or len(returns) < 2:
        return float("nan")
    q = 100.0 - confidence  # e.g. 95% -> 5th percentile
    pct = np.percentile(returns.values, q)
    return float(-pct)


def es_historical(returns: pd.Series, confidence: float = 95.0) -> float:
    """
    Historical Expected Shortfall (1-day): mean of daily returns in the
    worst (100 - confidence)% tail. Returned as positive loss.
    """
    if returns.empty or len(returns) < 2:
        return float("nan")
    q = 100.0 - confidence
    threshold = np.percentile(returns.values, q)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


def risk_contribution_percent(
    returns: pd.DataFrame,
    weights: dict,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Percent risk contribution by asset for a portfolio.

    Uses annualized covariance of daily returns:
    - cov_annual = cov_daily * trading_days_per_year
    - portfolio_vol = sqrt(w.T @ cov_annual @ w)
    - MCR = (cov_annual @ w) / portfolio_vol
    - risk_contribution = w * MCR (element-wise)
    - percent_contribution = risk_contribution / portfolio_vol  (sums to 1)

    Returns:
        pd.Series index=ticker, value=percent contribution (0 to 1).
    """
    if returns.empty or len(returns.columns) == 0 or len(returns) < 2:
        return pd.Series(dtype=float)

    cov_daily = returns.cov()
    cov_annual = cov_daily * trading_days_per_year
    w = np.array([weights.get(s, 0.0) for s in returns.columns], dtype=float)
    cov_arr = cov_annual.values
    sigma_p_sq = w @ cov_arr @ w
    if sigma_p_sq <= 0:
        return pd.Series(0.0, index=returns.columns)
    sigma_p = np.sqrt(sigma_p_sq)
    mcr = (cov_arr @ w) / sigma_p
    risk_contribution = w * mcr
    percent_contribution = risk_contribution / sigma_p
    return pd.Series(percent_contribution, index=returns.columns)


def portfolio_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Compute standard metrics for a single portfolio.
    Keys: total_return, cagr, volatility, sharpe_ratio, sortino_ratio, max_drawdown,
    var_95_1d, es_95_1d, var_99_1d, es_99_1d.
    """
    return {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve, trading_days_per_year),
        "volatility": annualized_volatility(returns, trading_days_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, trading_days_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, trading_days_per_year, mar=None),
        "max_drawdown": max_drawdown(equity_curve),
        "var_95_1d": var_historical(returns, confidence=95.0),
        "es_95_1d": es_historical(returns, confidence=95.0),
        "var_99_1d": var_historical(returns, confidence=99.0),
        "es_99_1d": es_historical(returns, confidence=99.0),
    }
