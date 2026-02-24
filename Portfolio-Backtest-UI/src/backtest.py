"""
Backtest engine: daily returns, rebalancing at month-end, transaction costs.
"""

from typing import Dict, Optional

import pandas as pd


def _last_day_of_month(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean series True on last trading day of each month."""
    if hasattr(dates, "tz") and dates.tz is not None:
        dates = dates.tz_localize(None)
    df = pd.DataFrame({"date": dates}, index=dates)
    df["month"] = df["date"].dt.to_period("M")
    last = df.groupby("month")["date"].transform("max")
    return df["date"] == last


def run_backtest(
    returns: pd.DataFrame,
    strategies: Dict[str, Dict[str, float]],
    rebalance_frequency: str = "monthly",
    transaction_cost_bps: float = 5.0,
    start_value: float = 100.0,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Run backtest for each strategy.

    Args:
        returns: DataFrame of daily asset returns (columns = symbols).
        strategies: Dict strategy_name -> {symbol: weight}. Weights should sum to 1.
        rebalance_frequency: "monthly" (month-end) or "none".
        transaction_cost_bps: Cost in bps per unit turnover on rebalance days.
        start_value: Starting portfolio value (equity curve indexed to this).

    Returns:
        Dict strategy_name -> {"equity_curve": pd.Series, "returns": pd.Series}.
    """
    if returns.empty or not list(returns.columns):
        return {}

    dates = returns.index
    is_rebalance_day = (
        _last_day_of_month(dates) if rebalance_frequency.lower() == "monthly" else pd.Series(False, index=dates)
    )
    cost_decimal_per_unit = transaction_cost_bps / 10_000.0

    result: Dict[str, Dict[str, pd.Series]] = {}

    for name, target_weights in strategies.items():
        w = pd.Series({s: target_weights.get(s, 0.0) for s in returns.columns})
        total_w = w.sum()
        if total_w == 0:
            continue
        w = w / total_w

        equity = pd.Series(index=dates, dtype=float)
        portfolio_returns = pd.Series(index=dates, dtype=float)

        prev_w = w.copy()
        equity.iloc[0] = start_value
        pr = (prev_w * returns.iloc[0]).sum()
        portfolio_returns.iloc[0] = 0.0 if pd.isna(pr) else float(pr)

        for i in range(1, len(dates)):
            if is_rebalance_day.iloc[i]:
                turnover = (w - prev_w).abs().sum() / 2.0
                cost = cost_decimal_per_unit * turnover
                pr = (w * returns.iloc[i]).sum()
                pr = 0.0 if pd.isna(pr) else float(pr)
                portfolio_returns.iloc[i] = pr - cost
                prev_w = w.copy()
            else:
                pr = (prev_w * returns.iloc[i]).sum()
                portfolio_returns.iloc[i] = 0.0 if pd.isna(pr) else float(pr)
            equity.iloc[i] = equity.iloc[i - 1] * (1 + portfolio_returns.iloc[i])

        result[name] = {"equity_curve": equity, "returns": portfolio_returns}

    return result


def equity_dataframe(results: Dict[str, Dict[str, pd.Series]]) -> pd.DataFrame:
    """Stack equity curves into a DataFrame (columns = portfolio names)."""
    return pd.DataFrame({name: data["equity_curve"] for name, data in results.items()})


def drawdowns_dataframe(results: Dict[str, Dict[str, pd.Series]]) -> pd.DataFrame:
    """Drawdown series per portfolio."""
    from .metrics import drawdown_series
    return pd.DataFrame({name: drawdown_series(data["equity_curve"]) for name, data in results.items()})
