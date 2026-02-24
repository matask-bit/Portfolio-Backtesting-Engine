# Portfolio Backtesting Engine

A Python project for backtesting multi-asset portfolios with configurable rebalancing and transaction costs. It downloads daily adjusted-close data, runs three portfolio strategies, computes standard performance metrics, and produces CSV outputs, charts, and an optional PDF report.

## Setup

1. **Clone or download** the project and open a terminal in the project root (`Portfolio-Backtesting-Engine/`).

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   # source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

- **Full pipeline** (download data → backtest → metrics → plots → CSV outputs):
  ```bash
  python scripts/run.py
  ```
  This writes:
  - `outputs/portfolio_metrics.csv` — one row per portfolio (Total Return, CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar).
  - `outputs/equity_curve.csv` — portfolio index values over time.
  - `outputs/drawdowns.csv` — drawdown series per portfolio.
  - `outputs/equity_curve.png`, `outputs/drawdown.png`, `outputs/rolling_vol.png`, `outputs/rolling_sharpe.png`.

- **PDF report** (run after the pipeline):
  ```bash
  python scripts/generate_report.py
  ```
  This creates `report/Portfolio_Backtest_Report.pdf` with a summary table, equity/drawdown charts, and a short interpretation section.

## Project Structure

```
Portfolio-Backtesting-Engine/
  scripts/
    config.py          # Assets, rebalance frequency, transaction cost, risk-free rate
    data.py            # Download (yfinance), clean, align prices; daily returns
    metrics.py         # Total return, CAGR, vol, Sharpe, Sortino, max DD, Calmar
    strategies.py      # Equal Weight, 60/40 Stock/Bond, SPY only
    backtest.py        # Backtest engine with rebalancing and transaction costs
    plots.py           # Equity curve, drawdown, rolling vol, rolling Sharpe
    run.py             # Full pipeline entry point
    generate_report.py # PDF report generation
  outputs/             # CSV and PNG outputs (created by run.py)
  report/              # PDF report (created by generate_report.py)
  README.md
  requirements.txt
```

## Assets and Data

- **Assets:** SPY, QQQ, MSFT, NVDA, TLT, GLD.
- **Data:** Adjusted close daily for the last 10 years (or max available). Missing data is cleaned and dates aligned across assets.

## Portfolios Simulated

1. **Equal Weight** — Equal weight across all 6 assets.
2. **60/40 Stock/Bond** — 60% stocks (SPY, QQQ, MSFT, NVDA equally within stock sleeve), 40% bonds (TLT).
3. **SPY Only** — 100% SPY.

## Configuration

Edit `scripts/config.py` to change:

- `REBALANCE_FREQUENCY`: `"monthly"` (end-of-month rebalance) or `"none"`.
- `TRANSACTION_COST_BPS`: e.g. `5` for 5 bps per unit turnover.
- `RISK_FREE_RATE`: annualized decimal (default `0`).
- `ASSETS`, `YEARS_BACK`, rolling windows for charts, etc.

## Requirements

- Python 3.10+ (for `list[str]` / `str | None` style hints; works with 3.9 if you change type hints).
- Libraries: pandas, numpy, matplotlib, yfinance, reportlab (optional for PDF).
