"""
Backtest engine: portfolio returns with rebalancing and transaction costs.

- Daily returns from adjusted close.
- Portfolio return = sum(weights * asset_returns).
- Rebalance monthly (end of month) or none; configurable.
- Transaction cost in bps per unit turnover (turnover = sum(|new_w - old_w|) / 2).
"""

from typing import Dict, Optional

import pandas as pd

from config import (
    REBALANCE_FREQUENCY,
    TRANSACTION_COST_BPS,
)
from strategies import get_all_strategies


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
    strategies: Optional[Dict[str, Dict[str, float]]] = None,
    rebalance_frequency: str = REBALANCE_FREQUENCY,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    start_value: float = 100.0,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Run backtest for each strategy.

    Args:
        returns: DataFrame of daily asset returns (columns = symbols).
        strategies: Dict strategy_name -> {symbol: weight}. Defaults to get_all_strategies().
        rebalance_frequency: "monthly" or "none".
        transaction_cost_bps: Cost in bps per unit turnover on rebalance.
        start_value: Starting portfolio value (for equity curve scale).

    Returns:
        Dict strategy_name -> {"equity_curve": pd.Series, "returns": pd.Series}.
    """
    strategies = strategies or get_all_strategies()
    # Align to common columns
    symbols = [c for c in returns.columns if c in returns.columns]
    if not symbols:
        raise ValueError("returns has no columns")

    dates = returns.index
    is_rebalance_day = _last_day_of_month(dates) if rebalance_frequency == "monthly" else pd.Series(False, index=dates)

    cost_decimal_per_unit = transaction_cost_bps / 10_000.0  # 5 bps = 0.0005

    result: Dict[str, Dict[str, pd.Series]] = {}

    for name, target_weights in strategies.items():
        # Ensure order and missing assets = 0
        w = pd.Series({s: target_weights.get(s, 0.0) for s in returns.columns})
        w = w / w.sum() if w.sum() != 0 else w

        equity = pd.Series(index=dates, dtype=float)
        portfolio_returns = pd.Series(index=dates, dtype=float)

        prev_w = w.copy()
        equity.iloc[0] = start_value
        # First day return: no rebalance
        pr = (prev_w * returns.iloc[0]).sum()
        if pd.isna(pr):
            pr = 0.0
        portfolio_returns.iloc[0] = pr
        # For equity: first row we set; then we'll compound
        for i in range(1, len(dates)):
            if is_rebalance_day.iloc[i]:
                turnover = (w - prev_w).abs().sum() / 2.0
                cost = cost_decimal_per_unit * turnover
                pr = (w * returns.iloc[i]).sum()
                if pd.isna(pr):
                    pr = 0.0
                portfolio_returns.iloc[i] = pr - cost
                prev_w = w.copy()
            else:
                # No rebalance: same weights as yesterday (buy-and-hold within day)
                pr = (prev_w * returns.iloc[i]).sum()
                if pd.isna(pr):
                    pr = 0.0
                portfolio_returns.iloc[i] = pr

            equity.iloc[i] = equity.iloc[i - 1] * (1 + portfolio_returns.iloc[i])

        result[name] = {"equity_curve": equity, "returns": portfolio_returns}

    return result


def backtest_results_to_equity_dataframe(
    results: Dict[str, Dict[str, pd.Series]],
) -> pd.DataFrame:
    """Stack equity curves into a DataFrame (columns = portfolio names)."""
    return pd.DataFrame({name: data["equity_curve"] for name, data in results.items()})


def backtest_results_to_drawdowns_dataframe(
    results: Dict[str, Dict[str, pd.Series]],
) -> pd.DataFrame:
    """Compute drawdown series for each portfolio and return as DataFrame."""
    from metrics import drawdown_series
    return pd.DataFrame({
        name: drawdown_series(data["equity_curve"]) for name, data in results.items()
    })
