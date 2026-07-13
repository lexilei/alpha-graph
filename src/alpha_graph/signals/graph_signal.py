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
        latest = latest.dropna(subset=["event_score"])  # NaN = no 8-K corpus
        return dict(zip(latest["ticker"], latest["event_score"]))

    if static_path.exists():
        df = pd.read_parquet(static_path)
        return dict(zip(df["ticker"], df["event_score"]))

    return {}


def _load_momentum(as_of_date: pd.Timestamp | None = None, horizon: int = 5,
                   market: pd.DataFrame | None = None) -> dict[str, float]:
    """Per-ticker `horizon`-day momentum as of a date: each ticker's LATEST
    non-NaN pct_change(horizon) at dates <= as_of_date (stale values carry
    forward for names that stopped trading). `market` is injectable for tests."""
    if market is None:
        market_path = CACHE_DIR / "market_data.parquet"
        if not market_path.exists():
            return {}
        market = pd.read_parquet(market_path)
    else:
        market = market.copy()
    market["date"] = pd.to_datetime(market["date"])

    if as_of_date is not None:
        market = market[market["date"] <= as_of_date]

    if market.empty:
        return {}

    market = market.sort_values(["ticker", "date"])
    market["mom"] = market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(horizon)
    )

    # Get latest per ticker
    latest = market.dropna(subset=["mom"]).groupby("ticker").tail(1)
    return dict(zip(latest["ticker"], latest["mom"]))


def customers_of(graph, ticker: str, min_conf: float = 0.8) -> list[tuple[str, float]]:
    """The firm's customers with edge confidence (Cohen-Frazzini direction).

    Customers = targets of out-edges labeled `customer` ("X is my customer")
    plus sources of in-edges labeled `supplier` ("ticker is X's supplier",
    i.e. X buys from ticker). Edges below min_conf are ignored.
    """
    out: list[tuple[str, float]] = []
    if not graph.has_node(ticker):
        return out
    for _, nb, d in graph.out_edges(ticker, data=True):
        if d.get("relation") == "customer" and d.get("confidence", 0.0) >= min_conf:
            out.append((nb, d["confidence"]))
    for nb, _, d in graph.in_edges(ticker, data=True):
        if d.get("relation") == "supplier" and d.get("confidence", 0.0) >= min_conf:
            out.append((nb, d["confidence"]))
    return out


def _customer_momentum_at(G, mom: dict[str, float], min_conf: float) -> list[dict]:
    """One grid date's rows: confidence-weighted mean of each node's
    customers' momentum. Shared by the monthly and daily builders so the
    construction is one piece of code (grid is a convention, not a variant)."""
    rows = []
    for ticker in G.nodes():
        cust = [(nb, c) for nb, c in customers_of(G, ticker, min_conf) if nb in mom]
        if cust:
            wsum = sum(c for _, c in cust)
            val = sum(mom[nb] * c for nb, c in cust) / wsum
        else:
            val = float("nan")
        rows.append({"ticker": ticker, "spillover_cust_mom": val})
    return rows


def compute_customer_momentum_timeseries(
    start_date: str | None = None,
    end_date: str | None = None,
    min_conf: float = 0.8,
    horizon: int = 21,
    relationships: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """C10 — confidence-weighted mean of the firm's CUSTOMERS' `horizon`-day
    momentum (the paper's customer→supplier lag, asymmetric); month-end grid.

    Saves CACHE_DIR / "graph_customer_momentum.parquet" (only when reading
    the real caches; injected relationships/market run cache-free for tests).
    """
    from alpha_graph.data.relationships import build_graph

    if start_date is None:
        start_date = "2012-06-30"
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    dates = pd.date_range(start=start_date, end=end_date, freq="ME")
    logger.info(f"Customer-momentum spillover for {len(dates)} month-ends "
                f"(conf>={min_conf}, horizon={horizon}d)...")

    rows = []
    for dt in dates:
        G = build_graph(relationships=relationships, as_of_date=dt)
        if G.number_of_edges() == 0:
            continue
        mom = _load_momentum(dt, horizon=horizon, market=market)
        for r in _customer_momentum_at(G, mom, min_conf):
            rows.append({**r, "date": dt})

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No customer-momentum rows computed")
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["spillover_cust_mom"] = df["spillover_cust_mom"].round(6)
    if relationships is None and market is None:
        out_path = CACHE_DIR / "graph_customer_momentum.parquet"
        df.to_parquet(out_path, index=False)
        n = df["spillover_cust_mom"].notna().sum()
        logger.info(f"Saved {len(df)} rows ({n} with qualifying customers) to {out_path}")
    return df


def compute_customer_momentum_daily(
    start_date: str = "2012-06-29",
    end_date: str | None = None,
    min_conf: float = 0.8,
    horizon: int = 21,
    relationships: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """C10 on the daily as-of grid — construction identical to the monthly
    builder (edges from filings <= t, customers' latest momentum <= t), grid
    is the only change, so it keeps the C10 registry ID. Rows are stamped
    with the computation date t; availability lag belongs to the panel merge
    (plan Item 2).

    The per-day loop loads everything once: momentum as a ffilled wide frame
    (identical "latest <= t" semantics to _load_momentum) and the edge list
    sorted by filing date, rebuilding the graph only on days the edge count
    changes. Saves graph_customer_momentum_daily.parquet on real runs.
    """
    from alpha_graph.data.relationships import build_graph

    if relationships is None:
        rel_path = CACHE_DIR / "relationships.parquet"
        if not rel_path.exists():
            logger.error("No relationships data. Run extraction first.")
            return pd.DataFrame()
        rel = pd.read_parquet(rel_path)
    else:
        rel = relationships.copy()
    rel["filing_date"] = pd.to_datetime(rel["filing_date"])
    rel = rel.sort_values("filing_date").reset_index(drop=True)
    fdates = rel["filing_date"].to_numpy()

    if market is None:
        market = pd.read_parquet(CACHE_DIR / "market_data.parquet")
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values(["ticker", "date"])
    market["mom"] = market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(horizon)
    )
    wide = (market.pivot_table(index="date", columns="ticker", values="mom",
                               aggfunc="last")
                  .sort_index().ffill())

    cal = wide.index
    cal = cal[cal >= pd.Timestamp(start_date)]
    if end_date is not None:
        cal = cal[cal <= pd.Timestamp(end_date)]
    logger.info(f"Customer-momentum spillover for {len(cal)} trading days "
                f"(conf>={min_conf}, horizon={horizon}d)...")

    rows = []
    G, prev_n = None, -1
    for dt in cal:
        n_edges = int(np.searchsorted(fdates, dt.to_datetime64(), side="right"))
        if n_edges == 0:
            continue
        if n_edges != prev_n:
            G = build_graph(relationships=rel.iloc[:n_edges])
            prev_n = n_edges
        mom = wide.loc[dt].dropna().to_dict()
        for r in _customer_momentum_at(G, mom, min_conf):
            rows.append({**r, "date": dt})

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No daily customer-momentum rows computed")
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["spillover_cust_mom"] = df["spillover_cust_mom"].round(6)
    df = df[["ticker", "date", "spillover_cust_mom"]]
    if relationships is None:
        out_path = CACHE_DIR / "graph_customer_momentum_daily.parquet"
        df.to_parquet(out_path, index=False)
        n = df["spillover_cust_mom"].notna().sum()
        logger.info(f"Saved {len(df)} daily rows ({n} with qualifying "
                    f"customers) to {out_path}")
    return df


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
    parser.add_argument(
        "--customer-momentum", action="store_true",
        help="C10: customers-only momentum spillover time series (monthly)"
    )
    parser.add_argument(
        "--daily", action="store_true",
        help="with --customer-momentum: daily as-of grid (plan Item 4)"
    )
    args = parser.parse_args()

    if args.customer_momentum:
        df = (compute_customer_momentum_daily() if args.daily
              else compute_customer_momentum_timeseries())
        if not df.empty:
            print(df.dropna().tail(10).to_string(index=False))
        return

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
