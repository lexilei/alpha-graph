"""Knowledge graph spillover signal — cross-firm information propagation.

Uses the inter-company relationship graph (supplier, customer, competitor,
partner) extracted from 10-K filings to propagate signals across firms.

When Company A files bad news (negative event_score), this signal predicts
the impact on its suppliers and customers before the market prices it in.

Edge-type weights reflect economic intuition:
  - supplier (1.0):   supply chain shocks propagate strongly to customers
  - customer (0.8):   customer distress affects upstream suppliers
  - partner (0.5):    partnership risk is shared
  - competitor (-0.3): competitor's bad news is weakly positive for you

The signal for each ticker is a confidence-weighted average of its
neighbors' event_score and momentum_5d.

Usage:
    python -m alpha_graph.signals.graph_signal [--as-of 2025-01-31]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR

# Edge-type weights for signal aggregation
EDGE_WEIGHTS: dict[str, float] = {
    "supplier": 1.0,    # strongest: supply chain disruption
    "customer": 0.8,    # strong: demand-side contagion
    "partner": 0.5,     # moderate: shared risk
    "competitor": -0.3,  # negative: competitor pain is your gain
}

# An edge (source -> target, rel) is stated from source's perspective:
# "target is source's {rel}". Seen from the TARGET, the relation flips:
# if I am your supplier, you are my customer.
REVERSED_RELATION: dict[str, str] = {
    "supplier": "customer",
    "customer": "supplier",
    "partner": "partner",
    "competitor": "competitor",
}


def compute_neighbor_signal(
    graph,
    ticker: str,
    signal_values: dict[str, float],
    edge_weights: dict[str, float] = EDGE_WEIGHTS,
) -> float:
    """Compute weighted average of neighbor signals for a single ticker.

    Considers both:
    - Out-edges: ticker -> target (ticker is supplier/partner to target)
    - In-edges:  source -> ticker (source is supplier/partner to ticker)

    Weight = edge_type_weight * confidence.
    Returns NaN if no neighbors have signal values — 0.0 is reserved for
    "neighbors exist and their weighted signal is genuinely zero"; conflating
    the two floods the cross-section with fake zeros for unconnected names.
    """
    weighted_sum = 0.0
    weight_total = 0.0

    # Out-edges: ticker -> neighbor
    if graph.has_node(ticker):
        for _, neighbor, data in graph.out_edges(ticker, data=True):
            if neighbor not in signal_values:
                continue
            rel = data.get("relation", "partner")
            conf = data.get("confidence", 0.5)
            w = edge_weights.get(rel, 0.0) * conf
            weighted_sum += w * signal_values[neighbor]
            weight_total += abs(w)

        # In-edges: neighbor -> ticker. The stored relation is from the
        # neighbor's perspective ("ticker is my {rel}"), so flip it to get
        # what the neighbor is to ticker before weighting.
        for neighbor, _, data in graph.in_edges(ticker, data=True):
            if neighbor not in signal_values:
                continue
            rel = REVERSED_RELATION.get(data.get("relation", "partner"), "partner")
            conf = data.get("confidence", 0.5)
            w = edge_weights.get(rel, 0.0) * conf
            weighted_sum += w * signal_values[neighbor]
            weight_total += abs(w)

    if weight_total == 0:
        return float("nan")

    return weighted_sum / weight_total


def _load_event_scores(as_of_date: pd.Timestamp | None = None) -> dict[str, float]:
    """Load per-ticker event scores, PIT when as_of_date is given.

    PIT mode never falls back to the static snapshot: that file is dateless
    (today's scores), and using it for a historical date propagates the
    present into the past.
    """
    ts_path = CACHE_DIR / "event_signals_timeseries.parquet"
    static_path = CACHE_DIR / "event_signals.parquet"

    if as_of_date is not None:
        if not ts_path.exists():
            logger.warning(
                "No event_signals_timeseries.parquet — spillover_event will be "
                "empty for PIT dates. Run: python -m alpha_graph.signals.event_signal --timeseries"
            )
            return {}
        df = pd.read_parquet(ts_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= as_of_date]
        if df.empty:
            return {}
        latest = df.sort_values("date").groupby("ticker").tail(1)
        return dict(zip(latest["ticker"], latest["event_score"]))

    if static_path.exists():
        df = pd.read_parquet(static_path)
        return dict(zip(df["ticker"], df["event_score"]))

    return {}


def _load_momentum(as_of_date: pd.Timestamp | None = None) -> dict[str, float]:
    """Load per-ticker 5-day momentum as of a given date."""
    market_path = CACHE_DIR / "market_data.parquet"
    if not market_path.exists():
        return {}

    market = pd.read_parquet(market_path)
    market["date"] = pd.to_datetime(market["date"])

    if as_of_date is not None:
        market = market[market["date"] <= as_of_date]

    if market.empty:
        return {}

    # Compute 5-day momentum
    market = market.sort_values(["ticker", "date"])
    market["mom_5d"] = market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(5)
    )

    # Get latest per ticker
    latest = market.dropna(subset=["mom_5d"]).groupby("ticker").tail(1)
    return dict(zip(latest["ticker"], latest["mom_5d"]))


def compute_spillover_signals(
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute spillover signals for all tickers as of a given date.

    Steps:
    1. Load relationships and build graph (respecting as_of_date)
    2. Load event_score and momentum_5d for all tickers
    3. For each ticker, compute neighbor-aggregated signals
    4. Return DataFrame with: ticker, date, spillover_event, spillover_momentum
    """
    from alpha_graph.data.relationships import build_graph

    if as_of_date is not None:
        as_of_date = pd.Timestamp(as_of_date)

    G = build_graph(as_of_date=as_of_date)
    if G.number_of_edges() == 0:
        logger.warning("Graph has no edges — cannot compute spillover")
        return pd.DataFrame()

    # Load signals
    event_scores = _load_event_scores(as_of_date)
    momentum_vals = _load_momentum(as_of_date)

    # Compute spillover for each ticker
    rows = []
    for ticker in G.nodes():
        spill_event = compute_neighbor_signal(G, ticker, event_scores)
        spill_mom = compute_neighbor_signal(G, ticker, momentum_vals)

        rows.append({
            "ticker": ticker,
            "date": as_of_date.strftime("%Y-%m-%d") if as_of_date else "",
            "spillover_event": round(float(spill_event), 6),
            "spillover_momentum": round(float(spill_mom), 6),
        })

    df = pd.DataFrame(rows)

    # NaN = no scored neighbors (unconnected); 0.0 = neighbors genuinely quiet
    has_signal = df["spillover_event"].notna() | df["spillover_momentum"].notna()
    n_connected = has_signal.sum()
    logger.info(
        f"Spillover signals: {n_connected}/{len(df)} tickers have graph neighbors"
    )

    return df


def compute_spillover_timeseries(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Compute spillover signal time series for walk-forward backtesting.

    For each month-end, rebuilds the graph using only filings filed
    before that date, then computes spillover signals.

    Returns DataFrame with: date, ticker, spillover_event, spillover_momentum.
    Saves to CACHE_DIR / "graph_spillover.parquet".
    """
    market_path = CACHE_DIR / "market_data.parquet"
    if not market_path.exists():
        logger.error("No market data. Run market download first.")
        return pd.DataFrame()

    market = pd.read_parquet(market_path)
    market["date"] = pd.to_datetime(market["date"])

    if start_date is None:
        start_date = "2023-06-30"
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    dates = pd.date_range(start=start_date, end=end_date, freq="ME")
    logger.info(f"Computing spillover time series for {len(dates)} month-ends...")

    all_dfs = []
    for dt in dates:
        df = compute_spillover_signals(as_of_date=dt)
        if not df.empty:
            df["date"] = dt.strftime("%Y-%m-%d")
            all_dfs.append(df)

    if not all_dfs:
        logger.warning("No spillover signals computed")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])

    # Save
    out_path = CACHE_DIR / "graph_spillover.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(result)} spillover observations to {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute knowledge graph spillover signals")
    parser.add_argument("--as-of", help="Compute as of date (YYYY-MM-DD)")
    parser.add_argument(
        "--timeseries", action="store_true",
        help="Compute full time series for backtesting"
    )
    args = parser.parse_args()

    if args.timeseries:
        df = compute_spillover_timeseries()
    else:
        df = compute_spillover_signals(as_of_date=args.as_of)

    if not df.empty:
        print("\n--- Knowledge Graph Spillover Signals ---")
        # Show connected tickers (NaN = no graph neighbors)
        has_signal = df["spillover_event"].notna() | df["spillover_momentum"].notna()
        active = df[has_signal].sort_values("spillover_event")

        if not active.empty:
            print(f"\nActive spillover signals ({len(active)} tickers):")
            print(active[["ticker", "spillover_event", "spillover_momentum"]].to_string())
        else:
            print("No active spillover signals (graph may have no edges)")

        # Graph summary
        from alpha_graph.data.relationships import build_graph, graph_summary
        G = build_graph(as_of_date=args.as_of)
        graph_summary(G)


if __name__ == "__main__":
    main()
