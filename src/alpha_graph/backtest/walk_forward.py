"""Walk-forward backtest for the Lazy Prices signal.

This is the core backtest that produces interview-credible metrics.
Uses historical Lazy Prices cosine similarity scores (no LLM calls needed)
across multiple rebalancing periods with purged walk-forward validation.

The signal: each time a company files a 10-K, compute cosine similarity
with the previous 10-K. Low similarity = large changes = SHORT.
High similarity = no changes = LONG. Rebalance monthly.

Usage:
    python -m alpha_graph.backtest.walk_forward
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from alpha_graph.backtest.engine import (
    BacktestConfig,
    build_long_short_portfolio,
    compute_performance_metrics,
    compute_portfolio_return,
    generate_tearsheet,
)
from alpha_graph.config import CACHE_DIR, FILINGS_DIR


def build_historical_signals() -> pd.DataFrame:
    """Build a panel of Lazy Prices signals with filing dates for walk-forward.

    For each ticker, computes cosine similarity between consecutive 10-K filings
    and assigns the signal as of the filing date. The signal persists until the
    next filing (stale signal = last known value).

    Returns DataFrame with: ticker, signal_date, cosine_similarity, signal
    """
    lazy_path = CACHE_DIR / "lazy_prices_signal_10K.parquet"
    if not lazy_path.exists():
        logger.error("Run lazy_prices signal first: python -m alpha_graph.signals.lazy_prices")
        return pd.DataFrame()

    signals = pd.read_parquet(lazy_path)
    signals = signals.rename(columns={"filing_date": "signal_date"})

    # signal is already computed: -1 (short), 0 (neutral), 1 (long)
    return signals[["ticker", "signal_date", "cosine_similarity", "signal"]].copy()


def build_monthly_signal_panel(
    signals: pd.DataFrame,
    market: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Expand point-in-time filing signals into a monthly panel.

    For each month-end, carry forward the most recent signal for each ticker.
    Then merge with forward returns from market data.

    Returns dict of {rebalance_date_str: DataFrame with ticker, signal, fwd_return}.
    """
    if signals.empty or market.empty:
        return {}

    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])

    # Get month-end dates from market data
    market["month_end"] = market["date"] + pd.offsets.MonthEnd(0)
    month_ends = sorted(market["month_end"].unique())

    panels = {}

    for me in month_ends:
        me_date = pd.Timestamp(me)

        # For each ticker, get the most recent signal as of this month-end
        valid_signals = signals[signals["signal_date"] <= me_date]
        if valid_signals.empty:
            continue

        latest_per_ticker = (
            valid_signals.sort_values("signal_date")
            .groupby("ticker")
            .tail(1)[["ticker", "signal", "cosine_similarity"]]
        )

        # Get forward 21-day returns from the closest trading day to month-end
        month_market = market[
            (market["date"] >= me_date - pd.Timedelta(days=5))
            & (market["date"] <= me_date)
        ]
        if month_market.empty:
            continue

        latest_prices = (
            month_market.sort_values("date")
            .groupby("ticker")
            .tail(1)[["ticker", "ret_21d"]]
            .rename(columns={"ret_21d": "fwd_return_21d"})
        )

        merged = latest_per_ticker.merge(latest_prices, on="ticker", how="inner")
        merged = merged.dropna(subset=["fwd_return_21d"])

        if len(merged) >= 10:  # need enough stocks
            # Map signal to combined_score for portfolio builder
            merged["combined_score"] = merged["signal"].astype(float)
            panels[me_date.strftime("%Y-%m-%d")] = merged

    return panels


def run_walk_forward() -> pd.DataFrame | None:
    """Execute the full walk-forward backtest."""
    # Load data
    signals = build_historical_signals()
    if signals.empty:
        return None

    market_path = CACHE_DIR / "market_data.parquet"
    if not market_path.exists():
        logger.error("Run market data download first")
        return None
    market = pd.read_parquet(market_path)

    # Build monthly panels
    panels = build_monthly_signal_panel(signals, market)
    logger.info(f"Built {len(panels)} monthly rebalancing periods")

    if not panels:
        logger.error("No valid rebalancing periods")
        return None

    # Run backtest across all periods
    config = BacktestConfig(top_n_long=10, top_n_short=10, cost_bps=10)
    results = []

    for date_str, panel in sorted(panels.items()):
        n_long = (panel["combined_score"] > 0).sum()
        n_short = (panel["combined_score"] < 0).sum()
        n_neutral = (panel["combined_score"] == 0).sum()

        # Adjust portfolio size based on available signals
        adj_long = min(config.top_n_long, max(n_long, 1))
        adj_short = min(config.top_n_short, max(n_short, 1))
        adj_config = BacktestConfig(
            top_n_long=adj_long, top_n_short=adj_short, cost_bps=config.cost_bps
        )

        portfolio = build_long_short_portfolio(panel, adj_config)
        period_result = compute_portfolio_return(portfolio, adj_config)
        period_result["date"] = date_str
        period_result["n_tickers_with_signal"] = len(panel)
        results.append(period_result)

    results_df = pd.DataFrame(results)
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df = results_df.sort_values("date").reset_index(drop=True)

    return results_df


def run_benchmark() -> pd.Series:
    """Get SPY monthly returns as benchmark."""
    market_path = CACHE_DIR / "market_data.parquet"
    if not market_path.exists():
        return pd.Series(dtype=float)

    # Download SPY separately
    market = pd.read_parquet(market_path)
    date_range = (market["date"].min(), market["date"].max())

    spy = yf.download(
        "SPY",
        start=date_range[0].strftime("%Y-%m-%d"),
        end=date_range[1].strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if spy.empty:
        return pd.Series(dtype=float)

    # Compute monthly returns
    spy_monthly = spy["Close"].resample("ME").last().pct_change().dropna()
    if hasattr(spy_monthly, "droplevel"):
        try:
            spy_monthly = spy_monthly.droplevel(1)
        except Exception:
            pass
    return spy_monthly


def main():
    results = run_walk_forward()
    if results is None:
        print("Walk-forward backtest failed — check data availability.")
        return

    # Performance metrics
    returns = results["net_return"]

    print("\n" + "=" * 60)
    print("  WALK-FORWARD BACKTEST: LAZY PRICES LONG-SHORT")
    print("=" * 60)

    # Per-period results
    print("\n--- Monthly Returns ---")
    for _, row in results.iterrows():
        print(
            f"  {row['date'].strftime('%Y-%m')}: "
            f"net={row['net_return']:+.2%}  "
            f"gross={row['gross_return']:+.2%}  "
            f"positions={row['n_positions']:.0f}  "
            f"tickers={row['n_tickers_with_signal']:.0f}"
        )

    # Tearsheet
    spy_returns = run_benchmark()
    # Align benchmark to same months as strategy
    if not spy_returns.empty:
        strategy_months = pd.to_datetime(results["date"]).dt.to_period("M")
        spy_monthly = spy_returns.copy()
        spy_monthly.index = spy_monthly.index.to_period("M")
        aligned_spy = spy_monthly[spy_monthly.index.isin(strategy_months)]
        benchmark = aligned_spy.reset_index(drop=True)
    else:
        benchmark = None

    print("\n" + generate_tearsheet(returns, benchmark))

    # Save results
    out_path = CACHE_DIR / "walk_forward_results.parquet"
    results.to_parquet(out_path, index=False)
    logger.info(f"Saved walk-forward results to {out_path}")


if __name__ == "__main__":
    main()
