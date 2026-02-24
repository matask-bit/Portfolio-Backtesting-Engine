"""
Performance metrics for portfolio backtesting.

Computes Total Return, CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar.
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def total_return(equity_curve: pd.Series) -> float:
    """Total return (decimal) from first to last value."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] == 0:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def cagr(equity_curve: pd.Series, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
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
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized Sharpe ratio (excess return / volatility).
    Risk-free rate is annualized; we convert to daily for mean.
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    rf_daily = risk_free_rate / trading_days_per_year
    excess = returns - rf_daily
    vol = returns.std() * np.sqrt(trading_days_per_year)
    if vol == 0:
        return 0.0
    return float(excess.mean() * trading_days_per_year / vol)


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    """Downside deviation (annualized): std of returns below MAR (e.g. 0)."""
    if returns.empty:
        return 0.0
    below = returns[returns < mar]
    if len(below) == 0:
        return 0.0
    return float(below.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    mar: Optional[float] = None,
) -> float:
    """
    Sortino ratio: (annualized excess return) / downside deviation.
    MAR defaults to risk_free_rate (annualized).
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    mar_ann = mar if mar is not None else risk_free_rate
    rf_daily = risk_free_rate / trading_days_per_year
    excess_ann = (returns.mean() - rf_daily) * trading_days_per_year
    dd = downside_deviation(returns, mar=mar_ann / trading_days_per_year)
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


def calmar_ratio(
    equity_curve: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calmar ratio = CAGR / abs(max drawdown)."""
    c = cagr(equity_curve, trading_days_per_year)
    md = max_drawdown(equity_curve)
    if md >= 0:
        return 0.0 if c <= 0 else float("inf")
    return float(c / abs(md))


def portfolio_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """
    Compute all standard metrics for a single portfolio.

    Returns:
        Dict with keys: total_return, cagr, volatility, sharpe_ratio,
        sortino_ratio, max_drawdown, calmar_ratio.
    """
    return {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve, trading_days_per_year),
        "volatility": annualized_volatility(returns, trading_days_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, trading_days_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, trading_days_per_year, mar=None),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(equity_curve, trading_days_per_year),
    }
