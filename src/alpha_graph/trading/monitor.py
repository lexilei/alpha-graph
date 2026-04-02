"""Paper trading monitor — track equity, positions, and P&L.

Provides a CLI dashboard for monitoring the Anti-Momentum paper trading strategy.
Reads from Alpaca API (if connected) and local trade logs.

Features:
  - Account summary (equity, cash, buying power)
  - Current positions with unrealized P&L
  - Trade history from local logs
  - Equity curve from Alpaca portfolio history
  - Signal summary from latest run

Usage:
    python -m alpha_graph.trading.monitor [--positions] [--history] [--signals]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, cfg


def show_account(client=None):
    """Display account summary."""
    if client is None:
        print("\n[No Alpaca connection — showing offline data only]")
        return

    from alpha_graph.trading.executor import get_account_info

    account = get_account_info(client)

    print("\n" + "=" * 50)
    print("  ACCOUNT SUMMARY")
    print("=" * 50)
    print(f"  Equity:        ${account['equity']:>12,.2f}")
    print(f"  Cash:          ${account['cash']:>12,.2f}")
    print(f"  Buying Power:  ${account['buying_power']:>12,.2f}")
    print(f"  Portfolio Val: ${account['portfolio_value']:>12,.2f}")


def show_positions(client=None):
    """Display current positions with P&L."""
    if client is None:
        print("\n  [No Alpaca connection]")
        return

    from alpha_graph.trading.executor import get_current_positions

    positions = get_current_positions(client)

    if not positions:
        print("\n  No open positions")
        return

    print("\n" + "=" * 70)
    print("  CURRENT POSITIONS")
    print("=" * 70)
    print(f"  {'Ticker':>8s}  {'Side':>6s}  {'Qty':>8s}  {'Mkt Value':>12s}  {'P&L':>10s}  {'P&L%':>8s}")
    print("  " + "-" * 62)

    total_value = 0
    total_pl = 0

    # Sort: longs first, then shorts
    sorted_positions = sorted(
        positions.items(),
        key=lambda x: (0 if x[1]["side"] == "long" else 1, x[0]),
    )

    for ticker, pos in sorted_positions:
        pl_color = "+" if pos["unrealized_pl"] >= 0 else ""
        print(
            f"  {ticker:>8s}  {pos['side']:>6s}  {pos['qty']:>8.1f}  "
            f"${abs(pos['market_value']):>11,.2f}  "
            f"{pl_color}${pos['unrealized_pl']:>9,.2f}  "
            f"{pos['unrealized_plpc']:>+7.2%}"
        )
        total_value += abs(pos["market_value"])
        total_pl += pos["unrealized_pl"]

    n_long = sum(1 for _, p in positions.items() if p["side"] == "long")
    n_short = sum(1 for _, p in positions.items() if p["side"] == "short")

    print("  " + "-" * 62)
    print(f"  {'TOTAL':>8s}  {f'{n_long}L/{n_short}S':>6s}  {'':>8s}  ${total_value:>11,.2f}  ${total_pl:>+9,.2f}")


def show_trade_history(n_days: int = 7):
    """Display recent trade history from local logs."""
    log_path = CACHE_DIR / "trade_log.csv"
    if not log_path.exists():
        print("\n  No trade history yet")
        return

    df = pd.read_csv(log_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter to recent
    cutoff = datetime.now() - timedelta(days=n_days)
    recent = df[df["timestamp"] >= cutoff]

    if recent.empty:
        print(f"\n  No trades in the last {n_days} days")
        return

    print(f"\n" + "=" * 80)
    print(f"  TRADE HISTORY (last {n_days} days)")
    print("=" * 80)
    print(f"  {'Date':>19s}  {'Action':>6s}  {'Side':>6s}  {'Ticker':>6s}  {'Value':>10s}  {'Signal':>8s}  {'Regime':>14s}")
    print("  " + "-" * 72)

    for _, row in recent.iterrows():
        dry_tag = " [DRY]" if row.get("dry_run", True) else ""
        print(
            f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M'):>19s}  "
            f"{row['action']:>6s}  {row['side']:>6s}  {row['ticker']:>6s}  "
            f"${row.get('value', 0):>9,.0f}  "
            f"{row.get('signal', 0):>+7.3f}  "
            f"{row.get('regime', ''):>14s}{dry_tag}"
        )

    print(f"\n  Total trades: {len(recent)}")


def show_signals():
    """Display latest Anti-Momentum signals."""
    signals_path = CACHE_DIR / "anti_momentum_signals.parquet"
    if not signals_path.exists():
        print("\n  No signals generated yet")
        return

    signals = pd.read_parquet(signals_path)

    print("\n" + "=" * 80)
    print("  LATEST ANTI-MOMENTUM SIGNALS")
    print("=" * 80)

    buys = signals[signals["recommendation"] == "BUY"]
    sells = signals[signals["recommendation"] == "SELL"]

    print(f"\n  BUY (Long) — {len(buys)} positions:")
    print(f"  {'Ticker':>8s}  {'Signal':>8s}  {'Pred Ret':>10s}  {'52w Dist':>10s}  {'Rev 5d':>10s}")
    print("  " + "-" * 50)
    for _, row in buys.iterrows():
        print(
            f"  {row['ticker']:>8s}  {row['signal']:>+7.3f}  "
            f"{row['predicted_return']:>+9.4f}  "
            f"{row['dist_52w_high']:>+9.2%}  "
            f"{row['reversal_5d']:>+9.2%}"
        )

    print(f"\n  SELL (Short) — {len(sells)} positions:")
    print(f"  {'Ticker':>8s}  {'Signal':>8s}  {'Pred Ret':>10s}  {'52w Dist':>10s}  {'Rev 5d':>10s}")
    print("  " + "-" * 50)
    for _, row in sells.iterrows():
        print(
            f"  {row['ticker']:>8s}  {row['signal']:>+7.3f}  "
            f"{row['predicted_return']:>+9.4f}  "
            f"{row['dist_52w_high']:>+9.2%}  "
            f"{row['reversal_5d']:>+9.2%}"
        )

    # Signal date
    if "date" in signals.columns:
        sig_date = pd.to_datetime(signals["date"].iloc[0]).strftime("%Y-%m-%d")
        print(f"\n  Signal date: {sig_date}")


def show_equity_curve(client=None):
    """Display portfolio equity curve from Alpaca."""
    if client is None:
        print("\n  [No Alpaca connection — cannot show equity curve]")
        return

    try:
        history = client.get_portfolio_history(
            period="1M",
            timeframe="1D",
        )

        if not history or not history.equity:
            print("\n  No portfolio history available")
            return

        print("\n" + "=" * 50)
        print("  EQUITY CURVE (Last 30 Days)")
        print("=" * 50)

        equities = history.equity
        timestamps = history.timestamp

        # Show last 10 data points
        n_show = min(10, len(equities))
        for i in range(-n_show, 0):
            ts = datetime.fromtimestamp(timestamps[i]).strftime("%Y-%m-%d")
            eq = equities[i]
            print(f"  {ts}  ${eq:>12,.2f}")

        if len(equities) >= 2:
            total_return = (equities[-1] / equities[0]) - 1
            print(f"\n  Period return: {total_return:+.2%}")

    except Exception as e:
        logger.debug(f"Failed to get portfolio history: {e}")
        print(f"\n  Could not retrieve equity curve: {e}")


def show_regime():
    """Display current market regime."""
    regime_path = CACHE_DIR / "regimes.parquet"
    if not regime_path.exists():
        print("\n  No regime data")
        return

    regimes = pd.read_parquet(regime_path)
    if regimes.empty:
        return

    latest = regimes.iloc[-1]
    print(f"\n  Current regime: {latest['regime']} (prob: {latest['regime_prob']:.2f})")
    print(f"  Long exposure:  {latest['long_exposure']:.0%}")
    print(f"  Short exposure: {latest['short_exposure']:.0%}")


def run_dashboard(
    show_all: bool = True,
    positions: bool = False,
    history: bool = False,
    signals: bool = False,
    equity: bool = False,
):
    """Run the monitoring dashboard."""
    # Try connecting to Alpaca
    client = None
    if cfg.alpaca_api_key and cfg.alpaca_secret_key:
        try:
            from alpha_graph.trading.executor import _get_trading_client
            client = _get_trading_client()
        except Exception as e:
            logger.debug(f"Alpaca connection failed: {e}")

    print("\n" + "█" * 50)
    print("  ANTI-MOMENTUM PAPER TRADING MONITOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 50)

    show_regime()

    if show_all or positions:
        show_account(client)
        show_positions(client)

    if show_all or signals:
        show_signals()

    if show_all or history:
        show_trade_history()

    if show_all or equity:
        show_equity_curve(client)


def main():
    parser = argparse.ArgumentParser(description="Anti-Momentum paper trading monitor")
    parser.add_argument("--positions", action="store_true", help="Show positions only")
    parser.add_argument("--history", action="store_true", help="Show trade history only")
    parser.add_argument("--signals", action="store_true", help="Show latest signals only")
    parser.add_argument("--equity", action="store_true", help="Show equity curve only")
    args = parser.parse_args()

    any_specific = args.positions or args.history or args.signals or args.equity

    run_dashboard(
        show_all=not any_specific,
        positions=args.positions,
        history=args.history,
        signals=args.signals,
        equity=args.equity,
    )


if __name__ == "__main__":
    main()
