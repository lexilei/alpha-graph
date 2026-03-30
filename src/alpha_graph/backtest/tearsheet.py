"""Generate quantstats HTML tearsheet from backtest results.

Produces a professional performance report with:
- Cumulative returns chart
- Drawdown visualization
- Monthly returns heatmap
- Rolling Sharpe ratio
- All standard risk metrics

Usage:
    python -m alpha_graph.backtest.tearsheet
"""

from __future__ import annotations

import pandas as pd
import quantstats as qs
from loguru import logger

from alpha_graph.config import CACHE_DIR, PROJECT_ROOT


def generate_html_tearsheet(
    results_path: str | None = None,
    output_name: str = "tearsheet",
    benchmark_ticker: str = "SPY",
) -> str:
    """Generate a quantstats HTML tearsheet from walk-forward results.

    Returns the path to the generated HTML file.
    """
    if results_path is None:
        results_path = CACHE_DIR / "walk_forward_results_regime.parquet"
        if not results_path.exists():
            results_path = CACHE_DIR / "walk_forward_results.parquet"

    if not results_path.exists():
        logger.error(f"No results at {results_path}. Run walk-forward backtest first.")
        return ""

    results = pd.read_parquet(results_path)
    results["date"] = pd.to_datetime(results["date"])

    # Build a returns Series with datetime index (quantstats needs this)
    returns = pd.Series(
        results["net_return"].values,
        index=results["date"],
        name="Strategy",
    )

    # Output path
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{output_name}.html"

    # Generate tearsheet
    logger.info(f"Generating tearsheet with benchmark={benchmark_ticker}...")
    qs.reports.html(
        returns,
        benchmark=benchmark_ticker,
        output=str(out_path),
        title="Alpha-Graph: Lazy Prices + HMM Regime Filter",
    )

    logger.info(f"Tearsheet saved to {out_path}")
    return str(out_path)


def main():
    # Generate both tearsheets
    raw_path = CACHE_DIR / "walk_forward_results.parquet"
    regime_path = CACHE_DIR / "walk_forward_results_regime.parquet"

    if raw_path.exists():
        path = generate_html_tearsheet(
            results_path=raw_path,
            output_name="tearsheet_no_filter",
        )
        if path:
            print(f"No-filter tearsheet: {path}")

    if regime_path.exists():
        path = generate_html_tearsheet(
            results_path=regime_path,
            output_name="tearsheet_regime_filter",
        )
        if path:
            print(f"Regime-filter tearsheet: {path}")


if __name__ == "__main__":
    main()
