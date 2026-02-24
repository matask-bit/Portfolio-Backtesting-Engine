"""
Full pipeline: download data → backtest → metrics → plots → CSV outputs.

Run from project root: python scripts/run.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Ensure scripts dir is on path when running from project root
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Project root (parent of scripts)
PROJECT_ROOT = os.path.dirname(_scripts_dir)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

from config import REBALANCE_FREQUENCY, RISK_FREE_RATE, TRANSACTION_COST_BPS

import pandas as pd
import data
import backtest
import metrics
import plots


def run_pipeline(
    outputs_dir: Optional[str] = None,
) -> dict:
    """
    Run full backtest pipeline and write CSVs + PNGs to outputs_dir.

    Returns:
        Dict with keys: prices, returns, results, equity_df, drawdowns_df, metrics_df, plot_paths.
    """
    outputs_dir = outputs_dir or os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # 1) Download and clean data
    print("Downloading and cleaning price data...")
    prices = data.load_data()
    returns = data.compute_returns(prices)
    returns = returns.dropna(how="all").dropna(axis=1, how="all")
    if returns.empty:
        raise RuntimeError("No returns after dropna; check data.")

    # 2) Backtest
    print("Running backtest...")
    results = backtest.run_backtest(
        returns,
        rebalance_frequency=REBALANCE_FREQUENCY,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )
    equity_df = backtest.backtest_results_to_equity_dataframe(results)
    drawdowns_df = backtest.backtest_results_to_drawdowns_dataframe(results)

    # 3) Metrics per portfolio
    print("Computing metrics...")
    rows = []
    for name, data_dict in results.items():
        m = metrics.portfolio_metrics(
            data_dict["equity_curve"],
            data_dict["returns"],
            risk_free_rate=RISK_FREE_RATE,
        )
        m["portfolio"] = name
        rows.append(m)
    metrics_df = pd.DataFrame(rows)
    # Reorder so portfolio is first
    cols = ["portfolio"] + [c for c in metrics_df.columns if c != "portfolio"]
    metrics_df = metrics_df[cols]

    # 4) Plots
    print("Generating plots...")
    plot_paths = plots.generate_all_plots(results, equity_df, output_dir=outputs_dir)

    # 5) Save CSVs
    metrics_path = os.path.join(outputs_dir, "portfolio_metrics.csv")
    equity_path = os.path.join(outputs_dir, "equity_curve.csv")
    drawdowns_path = os.path.join(outputs_dir, "drawdowns.csv")
    metrics_df.to_csv(metrics_path, index=False)
    equity_df.to_csv(equity_path)
    drawdowns_df.to_csv(drawdowns_path)
    print(f"Saved {metrics_path}, {equity_path}, {drawdowns_path}")

    return {
        "prices": prices,
        "returns": returns,
        "results": results,
        "equity_df": equity_df,
        "drawdowns_df": drawdowns_df,
        "metrics_df": metrics_df,
        "plot_paths": plot_paths,
    }


if __name__ == "__main__":
    try:
        run_pipeline(outputs_dir=os.path.join(PROJECT_ROOT, "outputs"))
        print("Pipeline finished successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        raise
