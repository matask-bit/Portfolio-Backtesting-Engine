# Portfolio-Backtest-UI

A Streamlit app for backtesting multi-asset portfolios with configurable tickers, date range, portfolio types, rebalancing, and transaction costs.

## Setup

1. Create a virtual environment (recommended) and activate it:

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate   # macOS/Linux
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run

From the project root:

```bash
streamlit run app.py
```

The app opens in your browser. Use the sidebar to set tickers, date range, portfolio type, rebalance frequency, transaction cost, and risk-free rate. Results include a metrics table, equity curve and drawdown charts, and a correlation heatmap. Use the download buttons to save `metrics.csv`, `equity_curve.csv`, and `drawdowns.csv`.

## Features

- **Tickers**: Comma-separated list (default: SPY, QQQ, MSFT, NVDA, TLT, GLD). Missing or invalid tickers are skipped; backtest uses only loaded symbols.
- **Date range**: Start/end dates; default last 10 years.
- **Portfolios**: Equal weight, 60/40 stock-bond (stocks = SPY, QQQ, MSFT, NVDA; bonds = TLT), SPY only, or custom weights (one weight per ticker, auto-normalized).
- **Rebalancing**: None or Monthly (month-end). When Monthly is selected, target weights are applied at the last trading day of each month.
- **Transaction costs**: Slider in bps per unit of turnover on rebalance days. Turnover = sum(|new weight − old weight|) / 2; cost = turnover × (bps / 10,000), subtracted from that day’s portfolio return.
- **Risk-free rate**: Used for Sharpe and Sortino (default 0%).

## Notes on transaction costs and rebalancing

- **Rebalancing** keeps the portfolio close to target weights. With “Monthly,” weights are reset at month-end; with “None,” the portfolio drifts (buy-and-hold).
- **Turnover** measures how much of the portfolio is traded: (1/2) × sum of absolute weight changes. Higher rebalance frequency or more divergent weights increase turnover.
- **Transaction cost** is applied only on rebalance days. Cost in basis points is per unit of turnover (e.g. 5 bps and 20% turnover → 1 bp drag on that day’s return). Default 5 bps is a common assumption for institutional trading.

## Project structure

```
Portfolio-Backtest-UI/
  app.py              # Streamlit entry point
  src/
    __init__.py
    data.py            # yfinance download, clean, returns (cache-friendly)
    portfolios.py      # Weight definitions (equal, 60/40, SPY, custom)
    backtest.py        # Backtest with rebalance and transaction costs
    metrics.py         # Total return, CAGR, vol, Sharpe, Sortino, max DD
    plots.py           # Equity, drawdown, correlation heatmap
    utils.py           # Parse tickers, normalize weights
  outputs/             # Optional local output folder
  requirements.txt
  README.md
```

## Requirements

- Python 3.8+
- streamlit, yfinance, pandas, numpy, matplotlib
