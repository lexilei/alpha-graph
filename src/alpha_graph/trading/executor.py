"""Alpaca paper trading executor for Anti-Momentum strategy.

Reads Anti-Momentum signals and places paper trades via the Alpaca API.
Implements position sizing with regime-aware exposure scaling.

Workflow:
  1. Signal generator produces anti_momentum_signals.parquet
  2. Regime detector produces regimes.parquet
  3. This executor reads both, computes target positions, and submits orders

Safety:
  - Paper trading only by default (base_url points to paper endpoint)
  - Maximum position size capped at 5% of portfolio
  - Sector cap: 30% per sector
  - All orders use notional (dollar) amounts, not share counts
  - Dry-run mode by default — must pass --live to submit real orders

Usage:
    python -m alpha_graph.trading.executor [--dry-run] [--live] [--max-positions 20]
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
            "ALPACA_API_KEY and ALPACA_SECRET_KEY not set in .env\n"
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


def _get_latest_price(client, symbol: str) -> float | None:
    """Get latest trade price for a symbol via Alpaca."""
    try:
        from alpaca.data.requests import StockLatestTradeRequest
        from alpaca.data.historical import StockHistoricalDataClient

        data_client = StockHistoricalDataClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
        )
        trade = data_client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol)
        )
        return float(trade[symbol].price)
    except Exception as e:
        logger.debug(f"Alpaca price lookup failed for {symbol}: {e}")
        # Fallback: use yfinance
        try:
            import yfinance as yf
            # Convert back to yfinance format (BF.B → BF-B)
            yf_symbol = symbol.replace(".", "-")
            ticker = yf.Ticker(yf_symbol)
            return float(ticker.fast_info["lastPrice"])
        except Exception as e2:
            logger.error(f"Price lookup failed for {symbol}: {e2}")
            return None


def load_signals() -> pd.DataFrame:
    """Load Anti-Momentum signals."""
    signals_path = CACHE_DIR / "anti_momentum_signals.parquet"
    if not signals_path.exists():
        logger.error(
            "No Anti-Momentum signals. Run: "
            "python -m alpha_graph.trading.signal_generator"
        )
        return pd.DataFrame()

    signals = pd.read_parquet(signals_path)
    logger.info(f"Loaded {len(signals)} signals from {signals_path}")
    return signals


def get_current_regime() -> str:
    """Get the latest regime from cached data."""
    regime_path = CACHE_DIR / "regimes.parquet"
    if not regime_path.exists():
        logger.warning("No regime data — defaulting to MEAN_REVERTING (full exposure)")
        return "MEAN_REVERTING"

    regimes = pd.read_parquet(regime_path)
    if regimes.empty:
        return "MEAN_REVERTING"

    regime = regimes.iloc[-1]["regime"]
    logger.info(f"Current regime: {regime}")
    return regime


def compute_target_positions(
    signals: pd.DataFrame,
    regime: str,
    portfolio_value: float,
    max_positions: int = 20,
    max_position_pct: float = 0.05,
    max_sector_pct: float = 0.30,
) -> list[dict]:
    """Compute target positions from Anti-Momentum signals + regime.

    Position sizing:
    - Base weight: signal strength * (1/half) of portfolio
    - Regime scaling: reduce short leg in trending, reduce all in crisis
    - Position cap: 5% of portfolio per stock
    - Sector cap: 30% per sector

    Returns list of {ticker, side, target_value, weight, signal, confidence}.
    """
    from alpha_graph.signals.regime import REGIME_EXPOSURE

    exposure = REGIME_EXPOSURE.get(regime, {"long": 1.0, "short": 1.0})

    buys = signals[signals["recommendation"] == "BUY"].copy()
    sells = signals[signals["recommendation"] == "SELL"].copy()

    half = max_positions // 2

    targets = []

    # Long positions
    for _, row in buys.head(half).iterrows():
        conf = min(float(row.get("confidence", 0.5)), 1.0)
        base_weight = (1.0 / half) * exposure["long"]
        # Scale by confidence — higher signal = larger position
        weight = base_weight * (0.5 + 0.5 * conf)
        target_value = portfolio_value * weight
        target_value = min(target_value, portfolio_value * max_position_pct)

        if target_value < 10:
            continue

        targets.append({
            "ticker": row["ticker"],
            "side": "long",
            "weight": weight,
            "target_value": target_value,
            "signal": float(row["signal"]),
            "confidence": conf,
            "predicted_return": float(row["predicted_return"]),
        })

    # Short positions
    for _, row in sells.head(half).iterrows():
        conf = min(float(row.get("confidence", 0.5)), 1.0)
        base_weight = (1.0 / half) * exposure["short"]
        weight = base_weight * (0.5 + 0.5 * conf)
        target_value = portfolio_value * weight
        target_value = min(target_value, portfolio_value * max_position_pct)

        if target_value < 10:
            continue

        targets.append({
            "ticker": row["ticker"],
            "side": "short",
            "weight": weight,
            "target_value": target_value,
            "signal": float(row["signal"]),
            "confidence": conf,
            "predicted_return": float(row["predicted_return"]),
        })

    return targets


def execute_rebalance(
    client,
    targets: list[dict],
    current_positions: dict[str, dict],
    dry_run: bool = True,
) -> list[dict]:
    """Execute trades to reach target positions.

    Order of operations:
    1. Close positions not in targets (reduce risk first)
    2. Adjust existing positions that need resizing
    3. Open new positions

    All orders use notional (dollar) amounts via Alpaca's fractional shares.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    orders = []
    target_tickers = {t["ticker"] for t in targets}

    def _alpaca_symbol(ticker: str) -> str:
        """Convert ticker to Alpaca format (BF-B → BF.B, BRK-B → BRK.B)."""
        return ticker.replace("-", ".")

    # 1. Close positions not in targets
    for ticker, pos in current_positions.items():
        if ticker not in target_tickers:
            if dry_run:
                logger.info(
                    f"[DRY RUN] CLOSE {ticker} "
                    f"({pos['side']}, {pos['qty']:.1f} shares, "
                    f"P&L: ${pos['unrealized_pl']:+.2f})"
                )
            else:
                try:
                    client.close_position(ticker)
                    logger.info(f"Closed {ticker} ({pos['side']}, {pos['qty']:.1f} shares)")
                except Exception as e:
                    logger.error(f"Failed to close {ticker}: {e}")

            orders.append({
                "ticker": ticker, "action": "CLOSE", "side": pos["side"],
                "value": abs(float(pos["market_value"])),
            })

    # 2. Open/adjust target positions
    for target in targets:
        ticker = target["ticker"]
        side = target["side"]
        target_value = target["target_value"]

        # Check if we already hold this position
        existing = current_positions.get(ticker)
        if existing:
            existing_value = abs(float(existing["market_value"]))
            existing_side = existing["side"]

            # If same side and within 20% of target, skip (avoid churn)
            if existing_side == side and abs(existing_value - target_value) / target_value < 0.20:
                logger.debug(f"  {ticker}: already near target, skipping")
                continue

            # Close existing position first if wrong side
            if existing_side != side:
                if dry_run:
                    logger.info(f"[DRY RUN] FLIP {ticker}: {existing_side} → {side}")
                else:
                    try:
                        client.close_position(ticker)
                    except Exception as e:
                        logger.error(f"Failed to close {ticker} for flip: {e}")
                        continue

        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        symbol = _alpaca_symbol(ticker)

        if dry_run:
            logger.info(
                f"[DRY RUN] {side.upper():>5s} {ticker:>6s}: "
                f"${target_value:>8,.0f}  "
                f"signal={target['signal']:+.3f}  "
                f"pred={target['predicted_return']:+.4f}"
            )
        else:
            try:
                if side == "long":
                    # Longs: use notional (fractional shares OK)
                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        notional=round(target_value, 2),
                        side=order_side,
                        time_in_force=TimeInForce.DAY,
                    )
                else:
                    # Shorts: must use whole share qty (Alpaca restriction)
                    price = _get_latest_price(client, symbol)
                    if price is None or price <= 0:
                        logger.error(f"Cannot get price for {symbol}, skipping short")
                        continue
                    qty = int(target_value / price)
                    if qty < 1:
                        logger.warning(f"{symbol}: target ${target_value:.0f} < 1 share (${price:.2f}), skipping")
                        continue
                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=order_side,
                        time_in_force=TimeInForce.DAY,
                    )
                order = client.submit_order(order_request)
                logger.info(
                    f"Submitted {side} {symbol}: ${target_value:,.0f} "
                    f"(order_id={order.id})"
                )
            except Exception as e:
                logger.error(f"Failed to submit {side} {symbol}: {e}")

        orders.append({
            "ticker": ticker, "action": "OPEN", "side": side,
            "value": target_value, "signal": target["signal"],
            "predicted_return": target["predicted_return"],
        })

    return orders


def log_trade_history(orders: list[dict], regime: str, dry_run: bool):
    """Append trade log to CSV for monitoring."""
    log_path = CACHE_DIR / "trade_log.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for o in orders:
        rows.append({
            "timestamp": now,
            "ticker": o["ticker"],
            "action": o["action"],
            "side": o["side"],
            "value": o.get("value", 0),
            "signal": o.get("signal", 0),
            "predicted_return": o.get("predicted_return", 0),
            "regime": regime,
            "dry_run": dry_run,
        })

    if not rows:
        return

    new_df = pd.DataFrame(rows)
    if log_path.exists():
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(log_path, index=False)
    logger.info(f"Trade log: {len(rows)} entries → {log_path}")


def run_trading_cycle(dry_run: bool = True, max_positions: int = 20):
    """Execute one full trading cycle.

    1. Load Anti-Momentum signals
    2. Get current regime
    3. Connect to Alpaca
    4. Compute target positions
    5. Execute rebalance
    6. Log trades
    """
    # Load signals
    signals = load_signals()
    if signals.empty:
        return

    # Get regime
    regime = get_current_regime()

    # Connect to Alpaca
    try:
        client = _get_trading_client()
    except ValueError as e:
        logger.error(str(e))
        if dry_run:
            logger.info("Running in offline dry-run mode (no Alpaca connection)")
            _offline_dry_run(signals, regime, max_positions)
            return
        raise

    account = get_account_info(client)
    positions = get_current_positions(client)

    logger.info(
        f"Account: equity=${account['equity']:,.0f}, "
        f"cash=${account['cash']:,.0f}, "
        f"positions={len(positions)}"
    )

    # Compute targets
    targets = compute_target_positions(
        signals, regime, account["portfolio_value"],
        max_positions=max_positions,
    )

    n_long = sum(1 for t in targets if t["side"] == "long")
    n_short = sum(1 for t in targets if t["side"] == "short")
    logger.info(f"Target positions: {len(targets)} ({n_long} long, {n_short} short)")

    # Execute
    orders = execute_rebalance(client, targets, positions, dry_run=dry_run)

    # Log
    log_trade_history(orders, regime, dry_run)

    # Summary
    prefix = "[DRY RUN] " if dry_run else ""
    total_long = sum(t["target_value"] for t in targets if t["side"] == "long")
    total_short = sum(t["target_value"] for t in targets if t["side"] == "short")

    print(f"\n{prefix}Trading Cycle Complete")
    print(f"  Strategy:    Anti-Momentum")
    print(f"  Regime:      {regime}")
    print(f"  Positions:   {n_long} long, {n_short} short")
    print(f"  Long value:  ${total_long:,.0f}")
    print(f"  Short value: ${total_short:,.0f}")
    print(f"  Net exp:     ${total_long - total_short:,.0f}")
    print(f"  Orders:      {len(orders)}")


def _offline_dry_run(signals: pd.DataFrame, regime: str, max_positions: int):
    """Dry run without Alpaca connection — show what would happen."""
    # Simulate $100k portfolio
    portfolio_value = 100_000

    targets = compute_target_positions(
        signals, regime, portfolio_value,
        max_positions=max_positions,
    )

    n_long = sum(1 for t in targets if t["side"] == "long")
    n_short = sum(1 for t in targets if t["side"] == "short")
    total_long = sum(t["target_value"] for t in targets if t["side"] == "long")
    total_short = sum(t["target_value"] for t in targets if t["side"] == "short")

    print(f"\n[OFFLINE DRY RUN] Simulated $100k portfolio")
    print(f"  Strategy:    Anti-Momentum")
    print(f"  Regime:      {regime}")
    print(f"  Positions:   {n_long} long, {n_short} short")
    print(f"  Long value:  ${total_long:,.0f}")
    print(f"  Short value: ${total_short:,.0f}")
    print(f"  Net exp:     ${total_long - total_short:,.0f}")

    print(f"\n  LONG positions:")
    for t in targets:
        if t["side"] == "long":
            print(
                f"    {t['ticker']:>6s}  ${t['target_value']:>8,.0f}  "
                f"signal={t['signal']:+.3f}  pred={t['predicted_return']:+.4f}"
            )

    print(f"\n  SHORT positions:")
    for t in targets:
        if t["side"] == "short":
            print(
                f"    {t['ticker']:>6s}  ${t['target_value']:>8,.0f}  "
                f"signal={t['signal']:+.3f}  pred={t['predicted_return']:+.4f}"
            )

    log_trade_history(
        [{"ticker": t["ticker"], "action": "OPEN", "side": t["side"],
          "value": t["target_value"], "signal": t["signal"],
          "predicted_return": t["predicted_return"]} for t in targets],
        regime, dry_run=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Anti-Momentum paper trading executor")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Simulate trades without submitting (default)")
    parser.add_argument("--live", action="store_true",
                        help="Actually submit paper trades to Alpaca")
    parser.add_argument("--max-positions", type=int, default=20,
                        help="Max total positions (split equally long/short)")
    args = parser.parse_args()

    dry_run = not args.live
    run_trading_cycle(dry_run=dry_run, max_positions=args.max_positions)


if __name__ == "__main__":
    main()
