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


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load signals and market data."""
    signals = build_historical_signals()
    if signals.empty:
        return None

    market_path = CACHE_DIR / "market_data.parquet"
    if not market_path.exists():
        logger.error("Run market data download first")
        return None
    market = pd.read_parquet(market_path)
    return signals, market


def _load_regimes() -> pd.DataFrame:
    """Load regime data, fitting HMM if needed."""
    regime_path = CACHE_DIR / "regimes.parquet"
    if regime_path.exists():
        return pd.read_parquet(regime_path)

    from alpha_graph.signals.regime import detect_regimes
    return detect_regimes()


def run_walk_forward(use_regime_filter: bool = False) -> pd.DataFrame | None:
    """Execute the full walk-forward backtest.

    Args:
        use_regime_filter: if True, scale position weights by HMM regime exposure.
    """
    data = _load_data()
    if data is None:
        return None
    signals, market = data

    # Build monthly panels
    panels = build_monthly_signal_panel(signals, market)
    logger.info(f"Built {len(panels)} monthly rebalancing periods")

    if not panels:
        logger.error("No valid rebalancing periods")
        return None

    # Load regimes if needed
    regimes = None
    if use_regime_filter:
        regimes = _load_regimes()
        if regimes.empty:
            logger.warning("No regime data — running without filter")
            use_regime_filter = False
        else:
            regimes["date"] = pd.to_datetime(regimes["date"])

    # Run backtest across all periods
    config = BacktestConfig(top_n_long=10, top_n_short=10, cost_bps=10)
    results = []

    for date_str, panel in sorted(panels.items()):
        n_long = (panel["combined_score"] > 0).sum()
        n_short = (panel["combined_score"] < 0).sum()

        # Adjust portfolio size based on available signals
        adj_long = min(config.top_n_long, max(n_long, 1))
        adj_short = min(config.top_n_short, max(n_short, 1))
        adj_config = BacktestConfig(
            top_n_long=adj_long, top_n_short=adj_short, cost_bps=config.cost_bps
        )

        portfolio = build_long_short_portfolio(panel, adj_config)

        # Apply regime filter: scale weights by exposure multipliers
        regime_label = "MEAN_REVERTING"  # default
        if use_regime_filter and regimes is not None:
            rebal_date = pd.Timestamp(date_str)
            # Get regime as of rebalance date (most recent available)
            past_regimes = regimes[regimes["date"] <= rebal_date]
            if not past_regimes.empty:
                latest = past_regimes.iloc[-1]
                regime_label = latest["regime"]
                long_mult = latest["long_exposure"]
                short_mult = latest["short_exposure"]

                portfolio.loc[portfolio["position"] == "LONG", "weight"] *= long_mult
                portfolio.loc[portfolio["position"] == "SHORT", "weight"] *= short_mult

        period_result = compute_portfolio_return(portfolio, adj_config)
        period_result["date"] = date_str
        period_result["n_tickers_with_signal"] = len(panel)
        period_result["regime"] = regime_label
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


def _print_results(results: pd.DataFrame, title: str, spy_returns: pd.Series):
    """Print backtest results with tearsheet."""
    returns = results["net_return"]

    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

    print("\n--- Monthly Returns ---")
    for _, row in results.iterrows():
        regime_str = f"  [{row['regime']}]" if "regime" in row and row.get("regime") else ""
        print(
            f"  {row['date'].strftime('%Y-%m')}: "
            f"net={row['net_return']:+.2%}  "
            f"gross={row['gross_return']:+.2%}  "
            f"pos={row['n_positions']:.0f}"
            f"{regime_str}"
        )

    # Align benchmark
    benchmark = None
    if not spy_returns.empty:
        strategy_months = pd.to_datetime(results["date"]).dt.to_period("M")
        spy_monthly = spy_returns.copy()
        spy_monthly.index = spy_monthly.index.to_period("M")
        aligned_spy = spy_monthly[spy_monthly.index.isin(strategy_months)]
        benchmark = aligned_spy.reset_index(drop=True)

    print("\n" + generate_tearsheet(returns, benchmark))


def main():
    # Run both versions
    results_raw = run_walk_forward(use_regime_filter=False)
    results_regime = run_walk_forward(use_regime_filter=True)

    if results_raw is None:
        print("Walk-forward backtest failed — check data availability.")
        return

    spy_returns = run_benchmark()

    _print_results(results_raw, "LAZY PRICES — NO REGIME FILTER", spy_returns)

    if results_regime is not None:
        _print_results(results_regime, "LAZY PRICES — WITH HMM REGIME FILTER", spy_returns)

        # Comparison
        raw_metrics = compute_performance_metrics(results_raw["net_return"])
        regime_metrics = compute_performance_metrics(results_regime["net_return"])
        print("\n" + "=" * 60)
        print("  REGIME FILTER IMPACT")
        print("=" * 60)
        print(f"  Sharpe:     {raw_metrics['sharpe']:+.3f}  ->  {regime_metrics['sharpe']:+.3f}")
        print(f"  Return:     {raw_metrics['total_return']:+.2%}  ->  {regime_metrics['total_return']:+.2%}")
        print(f"  Max DD:     {raw_metrics['max_drawdown']:.2%}  ->  {regime_metrics['max_drawdown']:.2%}")
        print(f"  % Positive: {raw_metrics['pct_positive']:.1%}  ->  {regime_metrics['pct_positive']:.1%}")
        print("=" * 60)

    # Save results
    out_path = CACHE_DIR / "walk_forward_results.parquet"
    results_raw.to_parquet(out_path, index=False)
    if results_regime is not None:
        out_path_regime = CACHE_DIR / "walk_forward_results_regime.parquet"
        results_regime.to_parquet(out_path_regime, index=False)
    logger.info(f"Saved walk-forward results to {CACHE_DIR}")


if __name__ == "__main__":
    main()
