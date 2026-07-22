"""First-cut latency analysis: how fast do Polymarket BTC-updown-5m quotes
react to Binance spot moves?

Method:
  1. Reconstruct per-token best bid/ask from the recorded CLOB stream
     (book snapshots + price_change deltas), restricted to btc-updown-5m
     tokens (front markets), using LOCAL receive timestamps.
  2. Binance BTC spot from the trade stream, same clock.
  3. Lead-lag: cross-correlate 100ms-grid diffs of spot vs poly mid.
  4. Jump study: for each spot move >= JUMP_BP within 1s, measure
     (a) poly mid before vs +5s after (the exploitable gap), and
     (b) time until the first poly quote revision after the jump.

Preliminary by design: no fair-value model yet — the +5s mid move is the
upper bound of what a sniper hitting stale quotes could capture, before
fees and queue.

Usage: .venv/bin/python scripts/polymarket_latency_analysis.py
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket"
JUMP_BP = 4.0


def load_lines(prefix: str):
    for f in sorted(RAW.glob(f"{prefix}_*.jsonl.gz")):
        try:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except EOFError:
            continue


def btc5m_tokens() -> dict[str, str]:
    toks = {}
    for d in load_lines("poly"):
        if d["src"] == "discovery":
            for tid, meta in d["msg"].items():
                if "btc-updown-5m" in meta.get("slug", ""):
                    toks[tid] = meta["slug"]
    return toks


def poly_mid_series(tokens: dict[str, str]) -> pd.DataFrame:
    books: dict[str, dict[str, dict[float, float]]] = defaultdict(
        lambda: {"bid": {}, "ask": {}})
    rows = []
    for d in load_lines("poly"):
        if d["src"] != "poly":
            continue
        msgs = d["msg"] if isinstance(d["msg"], list) else [d["msg"]]
        for m in msgs:
            aid = m.get("asset_id")
            if aid not in tokens:
                continue
            et = m.get("event_type")
            b = books[aid]
            if et == "book":
                b["bid"] = {float(x["price"]): float(x["size"])
                            for x in m.get("bids", [])}
                b["ask"] = {float(x["price"]): float(x["size"])
                            for x in m.get("asks", [])}
            elif et == "price_change":
                for ch in m.get("changes", []):
                    price, size = float(ch["price"]), float(ch["size"])
                    side = "bid" if ch["side"] == "BUY" else "ask"
                    if size == 0:
                        b[side].pop(price, None)
                    else:
                        b[side][price] = size
            else:
                continue
            bid = max(b["bid"]) if b["bid"] else np.nan
            ask = min(b["ask"]) if b["ask"] else np.nan
            if np.isfinite(bid) and np.isfinite(ask) and 0 < bid <= ask < 1:
                rows.append((d["t_local"], aid, (bid + ask) / 2, bid, ask))
    df = pd.DataFrame(rows, columns=["t", "token", "mid", "bid", "ask"])
    return df


def binance_series() -> pd.DataFrame:
    rows = []
    for d in load_lines("binance"):
        if d["src"] != "binance":
            continue
        m = d["msg"].get("data", {})
        if m.get("s") == "BTCUSDT":
            rows.append((d["t_local"], float(m["p"])))
    return pd.DataFrame(rows, columns=["t", "px"])


def main() -> None:
    tokens = btc5m_tokens()
    print(f"{len(tokens)} btc-updown-5m tokens seen")
    poly = poly_mid_series(tokens)
    spot = binance_series()
    print(f"poly quote updates: {len(poly):,} | binance trades: {len(spot):,}")

    # --- lead-lag on a 100ms grid, per token, averaged ---
    spot_g = (spot.assign(g=spot.t // 100_000).groupby("g").px.last())
    grid = np.arange(spot_g.index.min(), spot_g.index.max() + 1)
    spot_full = spot_g.reindex(grid).ffill()
    ccs = {}
    for tok, gdf in poly.groupby("token"):
        if len(gdf) < 300:
            continue
        pm = gdf.assign(g=gdf.t // 100_000).groupby("g").mid.last()
        pm = pm.reindex(grid).ffill()
        joint = pd.concat({"s": spot_full.diff(), "p": pm.diff()}, axis=1).dropna()
        joint = joint[(joint.s != 0) | (joint.p != 0)]
        if len(joint) < 200:
            continue
        for lag in range(0, 51, 2):  # 0..5s in 200ms steps
            c = joint.s.corr(joint.p.shift(-lag))
            ccs.setdefault(lag, []).append(c)
    lag_ms = {lag * 100: float(np.nanmean(v)) for lag, v in sorted(ccs.items())}
    best_lag = max(lag_ms, key=lambda k: lag_ms[k])
    print("\nlead-lag corr (spot diff -> poly mid diff, by poly delay):")
    for ms, c in lag_ms.items():
        bar = "#" * int(max(c, 0) * 200)
        print(f"  {ms:5d}ms  {c:+.3f} {bar}")
    print(f"peak correlation at poly delay ≈ {best_lag}ms")

    # --- jump study ---
    s = spot.copy()
    s["sec"] = s.t // 1_000_000
    sec_px = s.groupby("sec").px.last()
    jumps = (sec_px.diff().abs() / sec_px * 1e4)
    jump_secs = jumps[jumps >= JUMP_BP]
    print(f"\nspot jumps >= {JUMP_BP}bp/1s: {len(jump_secs)}")

    gaps, lat = [], []
    poly_sorted = poly.sort_values("t")
    pt = poly_sorted.t.values
    pmid = poly_sorted.mid.values
    for sec in jump_secs.index:
        t0 = int(sec) * 1_000_000
        i_pre = np.searchsorted(pt, t0) - 1
        i_post = np.searchsorted(pt, t0 + 5_000_000) - 1
        i_next = np.searchsorted(pt, t0)
        if i_pre < 0 or i_post <= i_pre or i_next >= len(pt):
            continue
        gaps.append(abs(pmid[i_post] - pmid[i_pre]))
        lat.append((pt[i_next] - t0) / 1000.0)
    gaps_s, lat_s = pd.Series(gaps), pd.Series(lat)
    print(f"poly mid |move| within 5s of a jump (cents): "
          f"p50 {gaps_s.quantile(0.5)*100:.1f} p75 {gaps_s.quantile(0.75)*100:.1f} "
          f"p90 {gaps_s.quantile(0.9)*100:.1f}")
    print(f"share of jumps where poly moved >= 3c (snipeable pre-fee): "
          f"{(gaps_s >= 0.03).mean():.1%}")
    print(f"time to FIRST poly quote update after jump (ms): "
          f"p10 {lat_s.quantile(0.1):.0f} p50 {lat_s.quantile(0.5):.0f} "
          f"p90 {lat_s.quantile(0.9):.0f}  "
          f"(floor = our own network RTT; treat as upper bound of staleness)")


if __name__ == "__main__":
    main()
