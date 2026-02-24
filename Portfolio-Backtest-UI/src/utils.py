"""
Utility helpers: parse tickers, normalize weights, constants.
"""

from typing import List

TRADING_DAYS_PER_YEAR = 252

# Default assets and classification for 60/40
DEFAULT_TICKERS = ["SPY", "QQQ", "MSFT", "NVDA", "TLT", "GLD"]
STOCK_TICKERS_60_40 = ["SPY", "QQQ", "MSFT", "NVDA"]
BOND_TICKERS_60_40 = ["TLT"]


def parse_tickers(text: str) -> List[str]:
    """
    Parse comma-separated tickers into a list of stripped, non-empty symbols.

    Args:
        text: User input string, e.g. "SPY, QQQ, MSFT".

    Returns:
        List of ticker strings. Empty list if text is empty or only commas/whitespace.
    """
    if not text or not isinstance(text, str):
        return []
    parts = [p.strip().upper() for p in text.split(",") if p.strip()]
    return parts


def normalize_weights(weights: dict) -> dict:
    """
    Normalize weight dict so values sum to 1.0.
    If sum is 0 or all zeros, returns weights unchanged (caller should handle).
    """
    total = sum(weights.values())
    if total is None or total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}
