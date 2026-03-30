"""Combined signal backtest — LLM pipeline + regime filter.

Unlike the walk-forward (which uses only Lazy Prices historically),
this evaluates the full multi-agent pipeline signals with regime-aware
position sizing against actual forward returns.

This is a single-period evaluation (not walk-forward) because we only
have one snapshot of LLM pipeline signals. In production, you'd run
the pipeline monthly and accumulate periods.

Usage:
    python -m alpha_graph.backtest.combined
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from alpha_graph.backtest.engine import (
    BacktestConfig,
    build_long_short_portfolio,
    compute_performance_metrics,
    compute_portfolio_return,
)
from alpha_graph.config import CACHE_DIR


def run_combined_backtest() -> None:
    """Run the combined LLM + regime backtest."""
    # Load pipeline signals
    signals_path = CACHE_DIR / "pipeline_signals.parquet"
    if not signals_path.exists():
        logger.error("No pipeline signals. Run: python -m alpha_graph.agents.pipeline")
        return
    signals = pd.read_parquet(signals_path)

    # Load market data for forward returns
    market_path = CACHE_DIR / "market_data.parquet"
    if not market_path.exists():
        logger.error("No market data. Run: python -m alpha_graph.data.market")
        return
    market = pd.read_parquet(market_path)

    # Get latest valid forward returns per ticker
    valid_market = market.dropna(subset=["ret_21d"])
    latest_prices = (
        valid_market.sort_values("date")
        .groupby("ticker")
        .tail(1)[["ticker", "date", "ret_21d"]]
        .rename(columns={"ret_21d": "fwd_return_21d", "date": "price_date"})
    )
    merged = signals.merge(latest_prices, on="ticker", how="inner")

    # Load regime
    regime_path = CACHE_DIR / "regimes.parquet"
    regime = "MEAN_REVERTING"
    if regime_path.exists():
        regimes = pd.read_parquet(regime_path)
        if not regimes.empty:
            regime = regimes.iloc[-1]["regime"]

    logger.info(f"Signals: {len(merged)} tickers, Regime: {regime}")

    # Run with different configs
    configs = [
        ("Full exposure (no filter)", BacktestConfig(top_n_long=10, top_n_short=10, cost_bps=10), False),
        ("Regime-filtered", BacktestConfig(top_n_long=10, top_n_short=10, cost_bps=10), True),
        ("Top 5 high-conviction", BacktestConfig(top_n_long=5, top_n_short=5, cost_bps=10), True),
    ]

    print("\n" + "=" * 70)
    print("  COMBINED SIGNAL BACKTEST: LLM PIPELINE + REGIME FILTER")
    print(f"  Regime: {regime} | Tickers: {len(merged)} | Period: ~21 trading days")
    print("=" * 70)

    for label, config, use_regime in configs:
        portfolio = build_long_short_portfolio(merged, config)

        # Apply regime scaling
        if use_regime:
            from alpha_graph.signals.regime import REGIME_EXPOSURE
            exposure = REGIME_EXPOSURE.get(regime, {"long": 1.0, "short": 1.0})
            portfolio.loc[portfolio["position"] == "LONG", "weight"] *= exposure["long"]
            portfolio.loc[portfolio["position"] == "SHORT", "weight"] *= exposure["short"]

        result = compute_portfolio_return(portfolio, config)

        # Breakdown
        longs = portfolio[portfolio["position"] == "LONG"]
        shorts = portfolio[portfolio["position"] == "SHORT"]
        long_ret = (longs["weight"] * longs["fwd_return_21d"].fillna(0)).sum()
        short_ret = (shorts["weight"] * shorts["fwd_return_21d"].fillna(0)).sum()

        print(f"\n  --- {label} ---")
        print(f"  Long leg ({len(longs)}):  {long_ret:+.2%}")
        print(f"  Short leg ({len(shorts)}): {short_ret:+.2%}")
        print(f"  Gross return:    {result['gross_return']:+.2%}")
        print(f"  Costs:           {result['total_cost']:.2%}")
        print(f"  Net return:      {result['net_return']:+.2%}")

        # Show top picks
        print(f"\n  Top longs:")
        for _, row in longs.head(5).iterrows():
            ret = row.get("fwd_return_21d", 0)
            print(f"    {row['ticker']:6s}  score={row['combined_score']:+.2f}  ret={ret:+.2%}")
        print(f"  Top shorts:")
        for _, row in shorts.head(5).iterrows():
            ret = row.get("fwd_return_21d", 0)
            print(f"    {row['ticker']:6s}  score={row['combined_score']:+.2f}  ret={ret:+.2%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_combined_backtest()
