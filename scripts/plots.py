"""
Chart generation: equity curve, drawdowns, rolling vol, rolling Sharpe.

Saves figures to outputs/.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUTS_DIR, ROLLING_SHARPE_WINDOW, ROLLING_VOL_WINDOW, TRADING_DAYS_PER_YEAR
from metrics import drawdown_series

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
except Exception:
    pass


def _ensure_output_dir() -> Path:
    p = Path(OUTPUTS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_equity_curve(
    equity_df: pd.DataFrame,
    output_dir: Optional[os.PathLike] = None,
    filename: str = "equity_curve.png",
) -> Path:
    """
    Plot all portfolios normalized to 100 on one chart.

    equity_df: columns = portfolio names, index = dates.
    """
    out = Path(output_dir or OUTPUTS_DIR)
    out.mkdir(parents=True, exist_ok=True)

    norm = equity_df / equity_df.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in norm.columns:
        ax.plot(norm.index, norm[col], label=col, linewidth=1.2)
    ax.set_title("Portfolio Equity Curves (Normalized to 100)")
    ax.set_ylabel("Index (100 = start)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(norm.index[0], norm.index[-1])
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_drawdown(
    equity_df: pd.DataFrame,
    output_dir: Optional[os.PathLike] = None,
    filename: str = "drawdown.png",
) -> Path:
    """Plot drawdown series for all portfolios."""
    out = Path(output_dir or OUTPUTS_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in equity_df.columns:
        dd = drawdown_series(equity_df[col])
        ax.fill_between(dd.index, dd, 0, alpha=0.5, label=col)
    ax.set_title("Portfolio Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rolling_vol(
    results: Dict[str, Dict[str, pd.Series]],
    window: int = ROLLING_VOL_WINDOW,
    output_dir: Optional[os.PathLike] = None,
    filename: str = "rolling_vol.png",
) -> Path:
    """63-day rolling annualized volatility per portfolio."""
    out = Path(output_dir or OUTPUTS_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in results.items():
        ret = data["returns"]
        roll_vol = ret.rolling(window, min_periods=window).std() * (TRADING_DAYS_PER_YEAR ** 0.5)
        roll_vol = roll_vol.dropna()
        if not roll_vol.empty:
            ax.plot(roll_vol.index, roll_vol, label=name, linewidth=1.0)
    ax.set_title(f"{window}-Day Rolling Annualized Volatility")
    ax.set_ylabel("Volatility (annualized)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rolling_sharpe(
    results: Dict[str, Dict[str, pd.Series]],
    window: int = ROLLING_SHARPE_WINDOW,
    risk_free_rate: float = 0.0,
    output_dir: Optional[os.PathLike] = None,
    filename: str = "rolling_sharpe.png",
) -> Path:
    """252-day rolling Sharpe ratio per portfolio."""
    from metrics import sharpe_ratio
    out = Path(output_dir or OUTPUTS_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in results.items():
        ret = data["returns"]
        roll_sharpe = ret.rolling(window, min_periods=window).apply(
            lambda x: sharpe_ratio(x.dropna(), risk_free_rate=risk_free_rate)
            if len(x.dropna()) >= window else float("nan"),
            raw=False,
        )
        roll_sharpe = roll_sharpe.dropna()
        if not roll_sharpe.empty:
            ax.plot(roll_sharpe.index, roll_sharpe, label=name, linewidth=1.0)
    ax.set_title(f"{window}-Day Rolling Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_plots(
    results: Dict[str, Dict[str, pd.Series]],
    equity_df: pd.DataFrame,
    output_dir: Optional[os.PathLike] = None,
) -> Dict[str, Path]:
    """
    Generate equity_curve, drawdown, rolling_vol, rolling_sharpe and save to output_dir.

    Returns:
        Dict plot_name -> path to saved file.
    """
    out = Path(output_dir or OUTPUTS_DIR)
    paths = {}
    paths["equity_curve"] = plot_equity_curve(equity_df, output_dir=out)
    paths["drawdown"] = plot_drawdown(equity_df, output_dir=out)
    paths["rolling_vol"] = plot_rolling_vol(results, output_dir=out)
    paths["rolling_sharpe"] = plot_rolling_sharpe(results, output_dir=out)
    return paths
