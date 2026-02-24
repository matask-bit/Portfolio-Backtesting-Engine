"""
Data download and preparation. Uses yfinance with caching and robust NaN handling.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf

from .utils import DEFAULT_TICKERS


def get_default_date_range(years_back: int = 10) -> Tuple[datetime, datetime]:
    """Return (start_date, end_date) for the last years_back years."""
    end = datetime.now()
    start = end - timedelta(days=years_back * 365)
    return start, end


def download_prices(
    symbols: List[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Download adjusted close daily prices for the given symbols.
    Skips symbols that fail or return no data; returns only available columns.

    Args:
        symbols: List of ticker symbols.
        start: Start date.
        end: End date.

    Returns:
        DataFrame with columns = successfully loaded symbols, index = DatetimeIndex.
        May be empty if all symbols fail.
    """
    if not symbols:
        return pd.DataFrame()

    all_data: List[pd.Series] = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start, end=end, auto_adjust=True)
            if hist.empty or "Close" not in hist.columns:
                continue
            close = hist["Close"].rename(sym)
            all_data.append(close)
        except Exception:
            continue

    if not all_data:
        return pd.DataFrame()
    prices = pd.concat(all_data, axis=1)
    prices = prices.reindex(sorted(prices.index.unique()))
    return prices


def clean_and_align(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing data: forward-fill, back-fill, then drop rows with any remaining NaN.
    Returns empty DataFrame if no common dates.
    """
    if prices.empty:
        return prices
    out = prices.ffill().bfill()
    out = out.dropna(how="any")
    return out


def load_prices(
    symbols: Optional[List[str]] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    years_back: int = 10,
) -> pd.DataFrame:
    """
    Download and return cleaned, aligned adjusted close prices.
    Uses default tickers and last 10 years if not specified.

    Returns:
        DataFrame with columns = symbols, index = DatetimeIndex.
    """
    symbols = symbols or DEFAULT_TICKERS
    if not start or not end:
        s, e = get_default_date_range(years_back)
        start = start or s
        end = end or e
    prices = download_prices(symbols, start, end)
    return clean_and_align(prices)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily (simple) returns from adjusted close. First row will be NaN."""
    if prices.empty:
        return prices
    return prices.pct_change()
