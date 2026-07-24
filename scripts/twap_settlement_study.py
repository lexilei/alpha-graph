"""Kalshi settlement mechanics study: BRTI basis + final-minute TWAP lock-in.

Kalshi BTC markets settle on the simple average of the 60 seconds of CF
Benchmarks' BRTI before the close. Two measurable questions on our recorded
data (brti_* = Coinbase+Kraken trades, poly-recorder binance stream, kalshi_*
ATM orderbooks):

  1. BASIS — how far does Binance trade price sit from the BRTI proxy
     (VW mid of Coinbase+Kraken trades, per second)? Mean/std/tails in bp.
     If material, every Binance-based FV (pin study, demo maker) carries it.
  2. TWAP LOCK-IN — during [T-60s, T], the settlement average accrues:
     settle = mean(BRTI over the window). At T-30s, half the average is
     locked. Quantify: |projected settle - naive spot| at T-45/-30/-15s in
     bp, and how often the lock flips the nearest strike's outcome
     probability by >20c vs a spot-only pricer.

Output: printed stats; no positions, measurement only.

Usage: .venv/bin/python scripts/twap_settlement_study.py
"""

from __future__ import annotations

import gzip

from gz_recover import iter_jsonl
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket"


def load_lines(prefix: str):
    # gz_recover: plain gzip readers crash on kill-corrupted members and
    # silently drop everything after them (nightly batch died on this)
    for f in sorted(RAW.glob(f"{prefix}_*.jsonl.gz")):
        yield from iter_jsonl(f)


def brti_proxy() -> pd.Series:
    rows = []
    for d in load_lines("brti"):
        t = d["t_local"] // 1_000_000
        m = d["msg"]
        if d["src"] == "coinbase":
            # ETH-USD joined the feed 07-24; this proxy is BTC only —
            # unfiltered, $1.8k ETH prints poison the VW mid
            if m.get("prod", "BTC-USD") != "BTC-USD":
                continue
            rows.append((t, float(m["p"]), float(m["s"])))
        elif d["src"] == "kraken":
            if isinstance(m, dict):  # post-ETH shape: {"pair", "trades"}
                if not str(m.get("pair", "XBT")).startswith("XBT"):
                    continue
                trades = m.get("trades") or []
            else:                    # legacy BTC-only shape: bare list
                trades = m
            for tr in trades:
                rows.append((t, float(tr[0]), float(tr[1])))
    df = pd.DataFrame(rows, columns=["t", "p", "v"])
    vw = df.groupby("t").apply(
        lambda g: np.average(g.p, weights=np.maximum(g.v, 1e-9)))
    full = np.arange(vw.index.min(), vw.index.max() + 1)
    return vw.reindex(full).ffill()


def binance_series() -> pd.Series:
    rows = []
    for d in load_lines("binance"):
        if d["src"] == "binance":
            m = d["msg"].get("data", {})
            if m.get("s") == "BTCUSDT":
                rows.append((d["t_local"] // 1_000_000, float(m["p"])))
    df = pd.DataFrame(rows, columns=["t", "p"]).groupby("t").p.last()
    return df


def main() -> None:
    brti = brti_proxy()
    bnc = binance_series()
    joint = pd.concat({"brti": brti, "bnc": bnc}, axis=1).dropna()
    basis_bp = (joint.bnc / joint.brti - 1) * 1e4
    print(f"coverage: {len(joint):,} joint seconds "
          f"({datetime.fromtimestamp(joint.index.min(), tz=timezone.utc):%m-%d %H:%M} "
          f".. {datetime.fromtimestamp(joint.index.max(), tz=timezone.utc):%m-%d %H:%M} UTC)")
    print(f"\nBASIS binance vs BRTI-proxy (bp): mean {basis_bp.mean():+.2f} "
          f"std {basis_bp.std():.2f} p1 {basis_bp.quantile(0.01):+.2f} "
          f"p99 {basis_bp.quantile(0.99):+.2f}")
    print(f"|basis| > 5bp share: {(basis_bp.abs() > 5).mean():.2%}")

    # hourly settlements covered: top of each UTC hour (ET tops == UTC tops)
    hours = [h for h in range(int(joint.index.min()) // 3600 * 3600 + 3600,
                              int(joint.index.max()), 3600)
             if h - 60 >= joint.index.min()]
    rows = []
    for H in hours:
        win = brti.loc[H - 60:H - 1]
        if win.isna().any() or len(win) < 60:
            continue
        settle = win.mean()
        for lead in (45, 30, 15):
            t0 = H - lead
            locked = brti.loc[H - 60:t0 - 1]
            spot = brti.loc[t0]
            proj = (locked.sum() + spot * lead) / 60  # naive projection
            rows.append({"H": H, "lead": lead,
                         "spot_vs_settle_bp": (spot / settle - 1) * 1e4,
                         "proj_vs_settle_bp": (proj / settle - 1) * 1e4})
    df = pd.DataFrame(rows)
    print(f"\nTWAP LOCK-IN ({df.H.nunique()} hourly settlements):")
    for lead in (45, 30, 15):
        g = df[df.lead == lead]
        print(f"  T-{lead:2d}s: |spot - settle| p50 "
              f"{g.spot_vs_settle_bp.abs().quantile(0.5):.2f}bp p90 "
              f"{g.spot_vs_settle_bp.abs().quantile(0.9):.2f}bp | "
              f"|TWAP-projection - settle| p50 "
              f"{g.proj_vs_settle_bp.abs().quantile(0.5):.2f}bp p90 "
              f"{g.proj_vs_settle_bp.abs().quantile(0.9):.2f}bp")
    print("\nreading: the projection error is the true residual risk in the "
          "final minute; the spot error is what a naive spot pricer carries. "
          "The gap between the two columns is the mechanical information the "
          "TWAP lock provides — free accuracy for any final-minute pricer.")


if __name__ == "__main__":
    main()
