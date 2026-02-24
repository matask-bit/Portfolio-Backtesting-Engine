"""
Matplotlib charts: equity curve, drawdown, correlation heatmap.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception:
            pass


def plot_equity_curve(
    equity_df: pd.DataFrame,
    title: str = "Equity Curve (Indexed to 100)",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot all portfolio equity curves normalized to 100. Returns matplotlib Figure."""
    _style()
    if equity_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        return fig
    norm = equity_df / equity_df.iloc[0] * 100
    fig, ax = plt.subplots(figsize=figsize)
    for col in norm.columns:
        ax.plot(norm.index, norm[col], label=col, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Index (100 = start)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(norm.index[0], norm.index[-1])
    fig.tight_layout()
    return fig


def plot_drawdown(
    equity_df: pd.DataFrame,
    title: str = "Drawdowns",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot drawdown series for each portfolio. Returns matplotlib Figure."""
    from .metrics import drawdown_series
    _style()
    fig, ax = plt.subplots(figsize=figsize)
    if equity_df.empty:
        ax.set_title(title)
        return fig
    for col in equity_df.columns:
        dd = drawdown_series(equity_df[col])
        ax.fill_between(dd.index, dd, 0, alpha=0.5, label=col)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Asset Return Correlation",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot correlation heatmap of asset returns. Returns matplotlib Figure."""
    _style()
    if returns.empty or len(returns.columns) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        return fig
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    return fig


def plot_rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 95.0,
    title: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Rolling 1-day VaR over time (historical method).
    For each date, VaR is computed over the previous `window` days.
    Returns matplotlib Figure.
    """
    _style()
    if returns.empty or len(returns) < window:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or f"Rolling {window}-Day VaR ({confidence}%)")
        return fig
    def var_single(x: pd.Series) -> float:
        if x.isna().all() or len(x) < 2:
            return np.nan
        q = 100.0 - confidence
        return -float(np.percentile(x.values, q))
    rolling_var = returns.rolling(window, min_periods=window).apply(var_single, raw=False)
    rolling_var = rolling_var.dropna()
    if rolling_var.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or f"Rolling {window}-Day VaR ({confidence}%)")
        return fig
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rolling_var.index, rolling_var.values, linewidth=1.2, color="tab:red")
    ax.set_title(title or f"Rolling {window}-Day VaR ({confidence}%)")
    ax.set_ylabel("VaR (1-day loss)")
    ax.set_xlabel("Date")
    ax.fill_between(rolling_var.index, rolling_var.values, 0, alpha=0.3, color="tab:red")
    fig.tight_layout()
    return fig


def plot_risk_contribution(
    percent_contribution: pd.Series,
    title: str = "Risk contribution by asset (%)",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Bar chart of percent risk contribution by ticker.
    percent_contribution: series index=ticker, value=contribution (0-1 or 0-100).
    """
    _style()
    if percent_contribution.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        return fig
    # Treat as decimal (0-1); display as percent
    pct = percent_contribution * 100.0
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(pct))
    bars = ax.bar(x, pct.values, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pct.index, rotation=45, ha="right")
    ax.set_ylabel("% of portfolio vol")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.8)
    fig.tight_layout()
    return fig
