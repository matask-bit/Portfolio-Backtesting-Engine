"""
Portfolio weight definitions: Equal Weight, 60/40, SPY only, Custom.
Adapts gracefully when tickers are missing.
"""

from typing import Dict, List

from .utils import BOND_TICKERS_60_40, STOCK_TICKERS_60_40


def weights_equal_weight(symbols: List[str]) -> Dict[str, float]:
    """Equal weight across all given symbols. Returns {} if symbols is empty."""
    if not symbols:
        return {}
    w = 1.0 / len(symbols)
    return {s: w for s in symbols}


def weights_60_40_stock_bond(symbols: List[str]) -> Dict[str, float]:
    """
    60% stocks (from STOCK_TICKERS_60_40 present in symbols, equally),
    40% bonds (from BOND_TICKERS_60_40 present in symbols).
    Missing tickers get 0; if no stocks or no bonds, allocates 100% to the available sleeve.
    """
    stock_syms = [s for s in symbols if s in STOCK_TICKERS_60_40]
    bond_syms = [s for s in symbols if s in BOND_TICKERS_60_40]
    out: Dict[str, float] = {}
    if stock_syms and bond_syms:
        for s in stock_syms:
            out[s] = 0.60 / len(stock_syms)
        for s in bond_syms:
            out[s] = 0.40 / len(bond_syms)
    elif stock_syms:
        for s in stock_syms:
            out[s] = 1.0 / len(stock_syms)
    elif bond_syms:
        for s in bond_syms:
            out[s] = 1.0 / len(bond_syms)
    for s in symbols:
        if s not in out:
            out[s] = 0.0
    return out


def weights_spy_only(symbols: List[str]) -> Dict[str, float]:
    """100% SPY if SPY in symbols; else 0 for all."""
    return {s: (1.0 if s == "SPY" else 0.0) for s in symbols}


def weights_custom(symbols: List[str], weight_by_ticker: Dict[str, float]) -> Dict[str, float]:
    """
    Use user-provided weights per ticker. Only symbols in weight_by_ticker get weight;
    missing symbols get 0. Weights are auto-normalized to sum to 1.0.
    """
    from .utils import normalize_weights
    w = {s: weight_by_ticker.get(s, 0.0) for s in symbols}
    return normalize_weights(w)
