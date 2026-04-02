"""Daily paper trading pipeline — single entry point.

Orchestrates the full Anti-Momentum trading workflow:
  1. Refresh market data (yfinance)
  2. Refresh 8-K event signals (SEC EDGAR filings)
  3. Retrain ML combiner (walk-forward LightGBM)
  4. Detect market regime (HMM)
  5. Generate Anti-Momentum signals
  6. Execute trades via Alpaca (paper)

Run daily at market close (4:30 PM ET) or before market open (9:00 AM ET).

Usage:
    # Full pipeline (refresh data + generate signals + trade)
    python -m alpha_graph.trading.daily_pipeline

    # Dry run (no actual trades)
    python -m alpha_graph.trading.daily_pipeline --dry-run

    # Live paper trades
    python -m alpha_graph.trading.daily_pipeline --live

    # Signals only (no trading)
    python -m alpha_graph.trading.daily_pipeline --signals-only

    # Skip data refresh (use cached data)
    python -m alpha_graph.trading.daily_pipeline --no-refresh
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from loguru import logger

from alpha_graph.config import CACHE_DIR


def setup_logging(log_dir: str | None = None):
    """Configure logging for daily pipeline runs."""
    if log_dir is None:
        log_dir = CACHE_DIR / "logs"
    log_dir = CACHE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{date_str}.log"

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
    )
    logger.info(f"Pipeline log: {log_file}")
    return log_file


def step_refresh_data(skip: bool = False):
    """Step 1: Refresh market data and event signals."""
    if skip:
        logger.info("[SKIP] Using cached data")
        return

    from alpha_graph.trading.signal_generator import (
        refresh_market_data,
        refresh_event_signals,
        refresh_regime,
    )

    refresh_market_data()
    refresh_event_signals()
    refresh_regime()


def step_retrain_ml_combiner(skip: bool = False):
    """Step 2: Retrain the base ML combiner (walk-forward)."""
    if skip:
        logger.info("[SKIP] Using cached ML combiner")
        return

    from alpha_graph.signals.ml_combiner import build_feature_panel, walk_forward_train_predict

    panel = build_feature_panel()
    if panel.empty:
        logger.error("Cannot build feature panel — aborting ML retrain")
        return

    results = walk_forward_train_predict(panel)
    if results.empty:
        logger.warning("ML combiner produced no results")


def step_generate_signals(top_n: int = 10):
    """Step 3: Generate Anti-Momentum signals."""
    from alpha_graph.trading.signal_generator import generate_signals

    signals = generate_signals(top_n=top_n, retrain=True)
    return signals


def step_execute_trades(dry_run: bool = True, max_positions: int = 20):
    """Step 4: Execute trades via Alpaca."""
    from alpha_graph.trading.executor import run_trading_cycle

    run_trading_cycle(dry_run=dry_run, max_positions=max_positions)


def run_pipeline(
    dry_run: bool = True,
    refresh: bool = True,
    retrain: bool = True,
    signals_only: bool = False,
    top_n: int = 10,
    max_positions: int = 20,
):
    """Run the full daily pipeline."""
    start = time.time()
    log_file = setup_logging()

    logger.info("=" * 60)
    logger.info("ANTI-MOMENTUM DAILY PIPELINE")
    logger.info(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Mode:       {'DRY RUN' if dry_run else 'LIVE PAPER TRADING'}")
    logger.info(f"  Refresh:    {'yes' if refresh else 'no (cached)'}")
    logger.info(f"  Retrain:    {'yes' if retrain else 'no (cached)'}")
    logger.info(f"  Top N:      {top_n}")
    logger.info(f"  Max pos:    {max_positions}")
    logger.info("=" * 60)

    try:
        # Step 1: Refresh data
        logger.info("--- Step 1/4: Refresh data ---")
        step_refresh_data(skip=not refresh)

        # Step 2: Retrain ML combiner
        logger.info("--- Step 2/4: Retrain ML combiner ---")
        step_retrain_ml_combiner(skip=not retrain)

        # Step 3: Generate signals
        logger.info("--- Step 3/4: Generate Anti-Momentum signals ---")
        signals = step_generate_signals(top_n=top_n)

        if signals is None or signals.empty:
            logger.error("No signals generated — aborting pipeline")
            return False

        if signals_only:
            logger.info("Signals-only mode — skipping trade execution")
            elapsed = time.time() - start
            logger.info(f"Pipeline complete in {elapsed:.1f}s")
            return True

        # Step 4: Execute trades
        logger.info("--- Step 4/4: Execute trades ---")
        step_execute_trades(dry_run=dry_run, max_positions=max_positions)

        elapsed = time.time() - start
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        print(f"\nPipeline completed in {elapsed:.1f}s. Log: {log_file}")
        return True

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        print(f"\nPipeline FAILED: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Anti-Momentum daily paper trading pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dry run (safe, no real trades)
  python -m alpha_graph.trading.daily_pipeline --dry-run

  # Live paper trading (submits orders to Alpaca paper account)
  python -m alpha_graph.trading.daily_pipeline --live

  # Just generate signals (no trades)
  python -m alpha_graph.trading.daily_pipeline --signals-only

  # Quick run with cached data
  python -m alpha_graph.trading.daily_pipeline --no-refresh --no-retrain
        """,
    )
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Simulate trades (default)")
    parser.add_argument("--live", action="store_true",
                        help="Submit paper trades to Alpaca")
    parser.add_argument("--no-refresh", action="store_true",
                        help="Skip data refresh, use cached data")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Skip ML combiner retrain, use cached model")
    parser.add_argument("--signals-only", action="store_true",
                        help="Generate signals without trading")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of long/short positions")
    parser.add_argument("--max-positions", type=int, default=20,
                        help="Max total positions")
    args = parser.parse_args()

    success = run_pipeline(
        dry_run=not args.live,
        refresh=not args.no_refresh,
        retrain=not args.no_retrain,
        signals_only=args.signals_only,
        top_n=args.top_n,
        max_positions=args.max_positions,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
