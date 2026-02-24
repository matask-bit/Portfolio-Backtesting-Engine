"""
Generate PDF report: summary table, equity/drawdown charts, interpretation.

Run after run.py. Reads from outputs/ and writes report/Portfolio_Backtest_Report.pdf.
"""

import os
import sys
from pathlib import Path
from typing import Optional

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
PROJECT_ROOT = os.path.dirname(_scripts_dir)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")

from config import REPORT_FILENAME

try:
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def _format_metrics_table(df: pd.DataFrame) -> list:
    """Build reportlab Table from metrics DataFrame."""
    cols = [c for c in df.columns]
    data = [cols]
    for _, row in df.iterrows():
        data.append([_fmt_cell(row[c], c) for c in cols])
    return data


def _fmt_cell(val, col: str) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, float):
        if "return" in col.lower() or "drawdown" in col.lower() or "volatility" in col.lower():
            return f"{val:.2%}"
        if "ratio" in col.lower() or "sharpe" in col.lower() or "sortino" in col.lower() or "calmar" in col.lower():
            return f"{val:.2f}"
        return f"{val:.4f}"
    return str(val)


def generate_report(
    outputs_dir: Optional[str] = None,
    report_dir: Optional[str] = None,
    report_filename: Optional[str] = None,
) -> str:
    """
    Create Portfolio_Backtest_Report.pdf from outputs/ CSVs and PNGs.

    Returns:
        Path to the created PDF file.
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab is required for PDF report. Install with: pip install reportlab")

    outputs_dir = outputs_dir or OUTPUTS_DIR
    report_dir = report_dir or REPORT_DIR
    report_filename = report_filename or REPORT_FILENAME
    os.makedirs(report_dir, exist_ok=True)
    pdf_path = os.path.join(report_dir, report_filename)

    metrics_path = os.path.join(outputs_dir, "portfolio_metrics.csv")
    equity_path = os.path.join(outputs_dir, "equity_curve.csv")
    drawdown_path = os.path.join(outputs_dir, "drawdown.png")
    equity_img_path = os.path.join(outputs_dir, "equity_curve.png")

    for p in [metrics_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Missing {p}. Run the full pipeline first: python scripts/run.py"
            )

    metrics_df = pd.read_csv(metrics_path)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Heading1"], fontSize=16, spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"], fontSize=12, spaceAfter=8,
    )
    body_style = styles["Normal"]

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )
    story = []

    story.append(Paragraph("Portfolio Backtest Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Summary table
    story.append(Paragraph("Summary: Portfolio Metrics", heading_style))
    table_data = _format_metrics_table(metrics_df)
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    # Equity curve chart
    if os.path.isfile(equity_img_path):
        story.append(Paragraph("Equity Curves (Normalized to 100)", heading_style))
        story.append(Image(equity_img_path, width=6 * inch, height=3 * inch))
        story.append(Spacer(1, 0.2 * inch))
    if os.path.isfile(drawdown_path):
        story.append(Paragraph("Drawdowns", heading_style))
        story.append(Image(drawdown_path, width=6 * inch, height=3 * inch))
        story.append(Spacer(1, 0.3 * inch))

    # Interpretation
    story.append(Paragraph("Interpretation", heading_style))
    interp = """
    This backtest compares three portfolios over the chosen period. 
    <b>Equal Weight</b> allocates the same weight to all six assets (SPY, QQQ, MSFT, NVDA, TLT, GLD), 
    which can lead to higher volatility because of the large weight in more volatile names (e.g. NVDA, MSFT). 
    <b>60/40 Stock/Bond</b> holds 60%% in stocks (SPY, QQQ, MSFT, NVDA equally within the stock sleeve) and 40%% in bonds (TLT). 
    Bonds typically reduce drawdowns and volatility, so this portfolio often shows lower maximum drawdown and 
    smoother equity curves than all-equity strategies. 
    <b>SPY Only</b> is a pure large-cap US equity benchmark. 
    Comparing risk and return across the three: diversification into bonds (60/40) usually reduces drawdowns; 
    equal weight may exhibit higher volatility due to concentrated exposure to high-growth names like NVDA and MSFT.
    """
    story.append(Paragraph(interp.strip(), body_style))

    doc.build(story)
    return pdf_path


if __name__ == "__main__":
    try:
        path = generate_report()
        print(f"Report saved to: {path}")
    except Exception as e:
        print(f"Report generation failed: {e}", file=sys.stderr)
        raise
