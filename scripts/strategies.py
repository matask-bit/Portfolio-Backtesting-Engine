"""
Portfolio strategy definitions (target weights).

Defines: Equal Weight, 60/40 Stock/Bond, and SPY-only.
"""

from typing import Dict

from config import ASSETS

# Stock tickers (for 60/40)
STOCKS = ["SPY", "QQQ", "MSFT", "NVDA"]
BONDS = ["TLT"]


def weights_equal_weight(symbols: list[str]) -> Dict[str, float]:
    """
    Equal weight across all given symbols.

    Args:
        symbols: List of ticker symbols (e.g. config.ASSETS).

    Returns:
        Dict mapping symbol -> weight (sum = 1.0).
    """
    if not symbols:
        return {}
    w = 1.0 / len(symbols)
    return {s: w for s in symbols}


def weights_60_40_stock_bond(symbols: list[str]) -> Dict[str, float]:
    """
    60% stocks (SPY, QQQ, MSFT, NVDA equally within stock sleeve), 40% bonds (TLT).

    Symbols not in STOCKS or BONDS get weight 0.
    """
    stock_syms = [s for s in symbols if s in STOCKS]
    bond_syms = [s for s in symbols if s in BONDS]

    out: Dict[str, float] = {}
    if stock_syms:
        w_stock_each = (0.60 / len(stock_syms))
        for s in stock_syms:
            out[s] = w_stock_each
    if bond_syms:
        w_bond_each = (0.40 / len(bond_syms))
        for s in bond_syms:
            out[s] = w_bond_each
    for s in symbols:
        if s not in out:
            out[s] = 0.0
    return out


def weights_spy_only(symbols: list[str]) -> Dict[str, float]:
    """100% SPY; all other symbols 0."""
    return {s: (1.0 if s == "SPY" else 0.0) for s in symbols}


def get_all_strategies() -> Dict[str, Dict[str, float]]:
    """
    Return dict of strategy_name -> target weights (for config.ASSETS).

    Used by backtest to run all portfolios.
    """
    symbols = list(ASSETS)
    return {
        "Equal_Weight": weights_equal_weight(symbols),
        "60_40_Stock_Bond": weights_60_40_stock_bond(symbols),
        "SPY_Only": weights_spy_only(symbols),
    }
