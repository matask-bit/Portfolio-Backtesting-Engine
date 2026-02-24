"""
Configuration for the Portfolio Backtesting Engine.

Centralizes assets, rebalance frequency, transaction costs, and risk-free rate.
"""

from typing import List

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
ASSETS: List[str] = ["SPY", "QQQ", "MSFT", "NVDA", "TLT", "GLD"]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
YEARS_BACK: int = 10  # Download last N years (or max available)

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
# Rebalance frequency: "monthly" (end of month) or "none"
REBALANCE_FREQUENCY: str = "monthly"

# Transaction cost in basis points per unit of turnover
# Turnover = sum(abs(new_w - old_w)) / 2
TRANSACTION_COST_BPS: float = 5.0

# Risk-free rate (annualized, decimal). Use 0.0 unless overridden.
RISK_FREE_RATE: float = 0.0

# ---------------------------------------------------------------------------
# Rolling windows for metrics/charts
# ---------------------------------------------------------------------------
ROLLING_VOL_WINDOW: int = 63   # Trading days
ROLLING_SHARPE_WINDOW: int = 252  # Trading days
TRADING_DAYS_PER_YEAR: int = 252

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
OUTPUTS_DIR: str = "outputs"
REPORT_DIR: str = "report"
REPORT_FILENAME: str = "Portfolio_Backtest_Report.pdf"
