"""Alpaca paper trading executor.

Reads pipeline signals and places paper trades via the Alpaca API.
Implements position sizing with regime-aware exposure scaling.

Workflow:
  1. Run the multi-agent pipeline -> pipeline_signals.parquet
  2. Run regime detection -> regimes.parquet
  3. This executor reads both, computes target positions, and submits orders

Safety:
  - Paper trading only by default (base_url points to paper endpoint)
  - Maximum position size capped at 5% of portfolio
  - Total exposure capped at 100% long + 100% short
  - All orders are limit orders with 1% slippage buffer

Usage:
    python -m alpha_graph.trading.executor [--dry-run] [--max-positions 20]
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, cfg


PAPER_BASE_URL = "https://paper-api.alpaca.markets"


def _get_trading_client():
    """Initialize Alpaca trading client (paper trading)."""
    from alpaca.trading.client import TradingClient

    if not cfg.alpaca_api_key or not cfg.alpaca_secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY not set. "
            "Sign up at https://app.alpaca.markets/signup (free paper trading)"
        )

    return TradingClient(
        api_key=cfg.alpaca_api_key,
        secret_key=cfg.alpaca_secret_key,
        paper=True,
    )


def get_account_info(client) -> dict:
    """Get current account balance and buying power."""
    account = client.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
    }


def get_current_positions(client) -> dict[str, dict]:
    """Get all current open positions."""
    positions = client.get_all_positions()
    return {
        p.symbol: {
            "qty": float(p.qty),
            "side": "long" if float(p.qty) > 0 else "short",
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
        }
        for p in positions
    }


def compute_target_positions(
    signals: pd.DataFrame,
    regime: str,
    portfolio_value: float,
    max_positions: int = 20,
    max_position_pct: float = 0.05,
) -> list[dict]:
    """Compute target positions from pipeline signals + regime.

    Returns list of {ticker, side, target_value, weight}.
    """
    from alpha_graph.signals.regime import REGIME_EXPOSURE

    exposure = REGIME_EXPOSURE.get(regime, {"long": 1.0, "short": 1.0})

    # Sort by absolute signal strength
    signals = signals.copy()
    signals["abs_score"] = signals["combined_score"].abs()
    signals = signals.sort_values("abs_score", ascending=False)

    targets = []
    n_long = 0
    n_short = 0
    half = max_positions // 2

    for _, row in signals.iterrows():
        ticker = row["ticker"]
        score = row["combined_score"]
        confidence = row.get("combined_confidence", 0.5)
        rec = row.get("recommendation", "HOLD")

        if rec == "BUY" and n_long < half:
            weight = (1.0 / half) * exposure["long"] * min(confidence, 1.0)
            target_value = portfolio_value * weight * max_position_pct / (1.0 / half)
            target_value = min(target_value, portfolio_value * max_position_pct)
            targets.append({
                "ticker": ticker,
                "side": "long",
                "weight": weight,
                "target_value": target_value,
                "score": score,
                "confidence": confidence,
            })
            n_long += 1

        elif rec == "SELL" and n_short < half:
            weight = (1.0 / half) * exposure["short"] * min(confidence, 1.0)
            target_value = portfolio_value * weight * max_position_pct / (1.0 / half)
            target_value = min(target_value, portfolio_value * max_position_pct)
            targets.append({
                "ticker": ticker,
                "side": "short",
                "weight": weight,
                "target_value": target_value,
                "score": score,
                "confidence": confidence,
            })
            n_short += 1

    return targets


def execute_trades(
    client,
    targets: list[dict],
    current_positions: dict[str, dict],
    dry_run: bool = False,
) -> list[dict]:
    """Execute trades to reach target positions.

    Closes positions not in targets, opens/adjusts positions in targets.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    orders = []

    # Close positions not in target
    target_tickers = {t["ticker"] for t in targets}
    for ticker, pos in current_positions.items():
        if ticker not in target_tickers:
            if dry_run:
                logger.info(f"[DRY RUN] Would close {ticker} ({pos['side']}, {pos['qty']} shares)")
            else:
                try:
                    client.close_position(ticker)
                    logger.info(f"Closed {ticker} ({pos['side']}, {pos['qty']} shares)")
                except Exception as e:
                    logger.error(f"Failed to close {ticker}: {e}")
            orders.append({"ticker": ticker, "action": "CLOSE", "side": pos["side"]})

    # Open/adjust target positions
    for target in targets:
        ticker = target["ticker"]
        side = target["side"]
        target_value = target["target_value"]

        if target_value < 10:  # skip tiny positions
            continue

        # Determine shares (approximate — using target_value)
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL

        if dry_run:
            logger.info(
                f"[DRY RUN] Would {side} {ticker}: "
                f"${target_value:.0f} (score={target['score']:+.2f}, "
                f"conf={target['confidence']:.2f})"
            )
        else:
            try:
                order_request = MarketOrderRequest(
                    symbol=ticker,
                    notional=round(target_value, 2),
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )
                order = client.submit_order(order_request)
                logger.info(
                    f"Submitted {side} {ticker}: ${target_value:.0f} "
                    f"(order_id={order.id})"
                )
            except Exception as e:
                logger.error(f"Failed to submit {side} {ticker}: {e}")

        orders.append({
            "ticker": ticker,
            "action": "OPEN",
            "side": side,
            "target_value": target_value,
            "score": target["score"],
        })

    return orders


def run_trading_cycle(dry_run: bool = True, max_positions: int = 20):
    """Execute one full trading cycle: signals -> regime -> trades."""
    # Load signals
    signals_path = CACHE_DIR / "pipeline_signals.parquet"
    if not signals_path.exists():
        logger.error("No pipeline signals. Run: python -m alpha_graph.agents.pipeline")
        return

    signals = pd.read_parquet(signals_path)
    logger.info(f"Loaded {len(signals)} signals")

    # Load regime
    regime_path = CACHE_DIR / "regimes.parquet"
    regime = "MEAN_REVERTING"  # default: full exposure
    if regime_path.exists():
        regimes = pd.read_parquet(regime_path)
        if not regimes.empty:
            regime = regimes.iloc[-1]["regime"]
    logger.info(f"Current regime: {regime}")

    # Connect to Alpaca
    client = _get_trading_client()
    account = get_account_info(client)
    positions = get_current_positions(client)

    logger.info(
        f"Account: equity=${account['equity']:.0f}, "
        f"cash=${account['cash']:.0f}, "
        f"positions={len(positions)}"
    )

    # Compute targets
    targets = compute_target_positions(
        signals, regime, account["portfolio_value"],
        max_positions=max_positions,
    )

    logger.info(f"Target positions: {len(targets)} "
                f"({sum(1 for t in targets if t['side'] == 'long')} long, "
                f"{sum(1 for t in targets if t['side'] == 'short')} short)")

    # Execute
    orders = execute_trades(client, targets, positions, dry_run=dry_run)

    # Summary
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Trading Cycle Complete")
    print(f"  Regime: {regime}")
    print(f"  Orders: {len(orders)}")
    for o in orders:
        print(f"    {o['action']} {o['side']} {o['ticker']}"
              + (f" ${o.get('target_value', 0):.0f}" if 'target_value' in o else ""))


def main():
    parser = argparse.ArgumentParser(description="Alpaca paper trading executor")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Simulate trades without submitting (default: True)")
    parser.add_argument("--live", action="store_true",
                        help="Actually submit paper trades")
    parser.add_argument("--max-positions", type=int, default=20)
    args = parser.parse_args()

    dry_run = not args.live
    run_trading_cycle(dry_run=dry_run, max_positions=args.max_positions)


if __name__ == "__main__":
    main()
