"""
Streamlit app: Portfolio Backtest UI.
Run with: streamlit run app.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Project root and src on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import drawdowns_dataframe, equity_dataframe, run_backtest
from src.data import compute_returns, load_prices
from src.metrics import portfolio_metrics, risk_contribution_percent
from src.plots import plot_correlation_heatmap, plot_drawdown, plot_equity_curve, plot_risk_contribution, plot_rolling_var
from src.portfolios import (
    weights_60_40_stock_bond,
    weights_custom,
    weights_equal_weight,
    weights_spy_only,
)
from src.utils import DEFAULT_TICKERS, parse_tickers


@st.cache_data(ttl=3600)
def cached_load_prices(
    tickers_str: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Load and clean price data. Cached by tickers and date range.
    Returns empty DataFrame on failure; missing tickers are skipped in data.py.
    """
    symbols = parse_tickers(tickers_str) if tickers_str else list(DEFAULT_TICKERS)
    if not symbols:
        return pd.DataFrame()
    try:
        return load_prices(symbols=symbols, start=start_date, end=end_date)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="Portfolio Backtest", layout="wide")
    st.title("Portfolio Backtest UI")

    # ---- Sidebar: inputs ----
    st.sidebar.header("Inputs")

    tickers_str = st.sidebar.text_input(
        "Tickers (comma-separated)",
        value=", ".join(DEFAULT_TICKERS),
        help="e.g. SPY, QQQ, MSFT, NVDA, TLT, GLD",
    )
    symbols = parse_tickers(tickers_str)
    if not symbols:
        st.warning("Enter at least one ticker.")
        return

    years_back = st.sidebar.slider("Years back (default 10)", 1, 20, 10)
    end_default = datetime.now()
    start_default = end_default - timedelta(days=years_back * 365)
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        start_date = st.date_input("Start date", value=start_default.date(), max_value=end_default.date())
    with col_d2:
        end_date = st.date_input("End date", value=end_default.date(), min_value=start_date)
    start_dt = datetime(start_date.year, start_date.month, start_date.day)
    end_dt = datetime(end_date.year, end_date.month, end_date.day)

    portfolio_type = st.sidebar.selectbox(
        "Portfolio",
        ["Equal Weight", "60/40 Stock-Bond", "SPY Only", "Custom Weights"],
        index=0,
    )

    custom_weights = {}
    if portfolio_type == "Custom Weights":
        st.sidebar.subheader("Weights (auto-normalized)")
        for s in symbols:
            custom_weights[s] = st.sidebar.number_input(f"{s} weight", min_value=0.0, max_value=1.0, value=1.0 / len(symbols) if symbols else 0.0, step=0.05, format="%.2f")

    rebalance = st.sidebar.selectbox("Rebalance frequency", ["None", "Monthly"], index=1)
    tc_bps = st.sidebar.slider("Transaction cost (bps per turnover)", 0, 50, 5, help="Applied on rebalance days")
    rf_pct = st.sidebar.number_input("Risk-free rate (%)", value=0.0, step=0.1, format="%.1f")
    risk_free_rate = rf_pct / 100.0

    # ---- Load data ----
    prices = cached_load_prices(tickers_str, start_dt, end_dt)
    if prices.empty or len(prices) < 2:
        st.error("No price data for the selected tickers and date range. Check tickers and try again.")
        return

    # Align symbols to what we actually have
    available = list(prices.columns)
    returns = compute_returns(prices).dropna(how="all")
    if returns.empty:
        st.error("Insufficient data to compute returns.")
        return

    # Build strategies dict (one or more portfolios)
    strategies = {}
    if portfolio_type == "Equal Weight":
        strategies["Equal Weight"] = weights_equal_weight(available)
    elif portfolio_type == "60/40 Stock-Bond":
        strategies["60/40 Stock-Bond"] = weights_60_40_stock_bond(available)
    elif portfolio_type == "SPY Only":
        strategies["SPY Only"] = weights_spy_only(available)
    else:
        strategies["Custom"] = weights_custom(available, custom_weights)

    # Drop strategies with zero total weight
    strategies = {k: v for k, v in strategies.items() if sum(v.values()) > 0}
    if not strategies:
        st.warning("No valid portfolio (e.g. SPY not in tickers for SPY Only, or custom weights sum to 0).")
        return

    rebalance_freq = "monthly" if rebalance == "Monthly" else "none"
    results = run_backtest(
        returns,
        strategies,
        rebalance_frequency=rebalance_freq,
        transaction_cost_bps=tc_bps,
        start_value=100.0,
    )
    if not results:
        st.error("Backtest produced no results.")
        return

    equity_df = equity_dataframe(results)
    drawdowns_df = drawdowns_dataframe(results)

    # Metrics table
    rows = []
    for name, data in results.items():
        m = portfolio_metrics(
            data["equity_curve"],
            data["returns"],
            risk_free_rate=risk_free_rate,
        )
        m["portfolio"] = name
        rows.append(m)
    metrics_df = pd.DataFrame(rows)
    cols_order = [
        "portfolio", "total_return", "cagr", "volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown",
        "var_95_1d", "es_95_1d", "var_99_1d", "es_99_1d",
    ]
    metrics_df = metrics_df[[c for c in cols_order if c in metrics_df.columns]]

    # ---- Main area: outputs ----
    st.subheader("Metrics")
    # Format for display (percent and decimals)
    display_df = metrics_df.copy()
    for col in ["total_return", "cagr", "volatility", "max_drawdown", "var_95_1d", "es_95_1d", "var_99_1d", "es_99_1d"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(x) else "")
    for col in ["sharpe_ratio", "sortino_ratio"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    st.dataframe(display_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Equity Curve")
        fig_eq = plot_equity_curve(equity_df)
        st.pyplot(fig_eq)
        plt.close(fig_eq)
    with col2:
        st.subheader("Drawdowns")
        fig_dd = plot_drawdown(equity_df)
        st.pyplot(fig_dd)
        plt.close(fig_dd)

    st.subheader("Asset Return Correlation")
    fig_corr = plot_correlation_heatmap(returns)
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    # Rolling VaR (252-day VaR 95%) for selected portfolio
    st.subheader("Rolling VaR (95%)")
    portfolio_for_var = st.selectbox(
        "Portfolio for rolling 252-day VaR",
        options=list(results.keys()),
        index=0,
        key="rolling_var_portfolio",
    )
    port_returns = results[portfolio_for_var]["returns"]
    fig_var = plot_rolling_var(port_returns, window=252, confidence=95.0)
    st.pyplot(fig_var)
    plt.close(fig_var)

    # Risk contribution by asset for selected portfolio
    st.subheader("Risk contribution by asset")
    portfolio_for_risk = st.selectbox(
        "Portfolio for risk contribution",
        options=list(results.keys()),
        index=0,
        key="risk_contrib_portfolio",
    )
    weights_selected = strategies[portfolio_for_risk]
    risk_pct = risk_contribution_percent(returns, weights_selected)
    fig_risk = plot_risk_contribution(risk_pct, title="Risk contribution by asset (%)")
    st.pyplot(fig_risk)
    plt.close(fig_risk)

    # Download buttons
    st.subheader("Download")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download metrics.csv",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="metrics.csv",
            mime="text/csv",
            key="dl_metrics",
        )
    with c2:
        st.download_button(
            "Download equity_curve.csv",
            data=equity_df.to_csv().encode("utf-8"),
            file_name="equity_curve.csv",
            mime="text/csv",
            key="dl_equity",
        )
    with c3:
        st.download_button(
            "Download drawdowns.csv",
            data=drawdowns_df.to_csv().encode("utf-8"),
            file_name="drawdowns.csv",
            mime="text/csv",
            key="dl_drawdowns",
        )


if __name__ == "__main__":
    main()
