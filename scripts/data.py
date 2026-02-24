"""
Data download and preparation for the Portfolio Backtesting Engine.

Uses yfinance to fetch adjusted close daily data, cleans missing values,
and aligns dates across all assets.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import ASSETS, YEARS_BACK


def get_date_range() -> Tuple[datetime, datetime]:
    """Return (start_date, end_date) for the requested lookback period."""
    end = datetime.now()
    start = end - timedelta(days=YEARS_BACK * 365)
    return start, end


def download_prices(
    symbols: Optional[list[str]] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Download adjusted close daily prices for the given symbols.

    Args:
        symbols: List of ticker symbols. Defaults to config.ASSETS.
        start: Start date. Defaults to YEARS_BACK from today.
        end: End date. Defaults to today.

    Returns:
        DataFrame with columns = symbols, index = DatetimeIndex (dates).
        Uses 'Adj Close' from yfinance.
    """
    symbols = symbols or ASSETS
    if not start or not end:
        s, e = get_date_range()
        start = start or s
        end = end or e

    all_data: list[pd.Series] = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start, end=end, auto_adjust=True)
            if hist.empty:
                raise ValueError(f"No data returned for {sym}")
            # yfinance with auto_adjust=True uses "Close" as adjusted
            close = hist["Close"].rename(sym)
            all_data.append(close)
        except Exception as e:
            raise RuntimeError(f"Failed to download {sym}: {e}") from e

    # Align all series on common index (union of dates, then forward-fill then dropna for leading NaNs)
    prices = pd.concat(all_data, axis=1)
    prices = prices.reindex(sorted(prices.index.unique()))

    return prices


def clean_and_align(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing data and align dates across assets.

    - Forward-fill then back-fill limited gaps.
    - Drop rows where any asset still has NaN (e.g. different listing dates).
    """
    out = prices.ffill().bfill()
    # Drop any row that still has NaN (e.g. first days before any data)
    out = out.dropna(how="any")
    if out.empty:
        raise ValueError("No common dates after cleaning; check data availability.")
    return out


def load_data(
    symbols: Optional[list[str]] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Download and return cleaned, aligned adjusted close prices.

    Returns:
        DataFrame with columns = symbols, index = DatetimeIndex.
    """
    prices = download_prices(symbols=symbols, start=start, end=end)
    return clean_and_align(prices)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily (simple) returns from adjusted close prices.

    Returns:
        DataFrame of daily returns, same shape as prices (first row NaN).
    """
    return prices.pct_change()


if __name__ == "__main__":
    # Quick test
    df = load_data()
    print(df.head())
    print(df.tail())
    print("Shape:", df.shape)
