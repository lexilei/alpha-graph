"""Markout / adverse-selection read on the live maker's fills.

For each session fill, recompute the maker's own fair value at the fill
instant and at +30s / +2min using the recorded spot trajectory (same
N(log(S/K)/(sigma*sqrt(tau))) the maker quotes on), sign it by our
position direction, and average. Negative = adverse (we were picked off).

This is the number that decides kill vs scale — but it is only meaningful
at scale. On a tiny same-session sample it is a PLUMBING CHECK: does the
telemetry capture enough to compute markout, and is the sign sane.

Usage: .venv/bin/python scripts/markout_analysis.py [cutover_iso]
  cutover_iso: only fills created at/after this UTC time (default: today
  16:57Z, the prod cutover) — excludes the 07-22 manual test fills.
"""

from __future__ import annotations

import glob
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gz_recover import iter_jsonl
import kalshi_client as kc

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "polymarket"


def load_maker(env: str = "prod"):
    fills, spots = [], []
    for f in sorted(glob.glob(str(RAW / f"maker{env}_*.jsonl.gz"))):
        for d in iter_jsonl(Path(f)):
            if d["src"] == "fill":
                fills.append(d)
            elif d["src"] == "maker_cycle":
                m = d["msg"]
                if m.get("spot"):
                    spots.append((d["t_local"], float(m["spot"])))
    spots.sort()
    return fills, spots


def spot_at(spots, t_us):
    """Linear-interp the maker's spot series at a t_local microsecond."""
    ts = np.array([s[0] for s in spots])
    px = np.array([s[1] for s in spots])
    if t_us <= ts[0] or t_us >= ts[-1]:
        return None
    return float(np.interp(t_us, ts, px))


_mkt_cache: dict = {}


def market_meta(k, ticker):
    if ticker in _mkt_cache:
        return _mkt_cache[ticker]
    try:
        m = k.get(f"/markets/{ticker}")["market"]
        meta = (m.get("floor_strike"), m.get("cap_strike"),
                m.get("close_time"))
    except Exception:  # noqa: BLE001
        meta = (None, None, None)
    _mkt_cache[ticker] = meta
    return meta


def fv(spot, floor, cap, sigma, tau_s):
    """The maker's own fair value: P(S>floor) minus P(S>cap) for a bracket."""
    if not (spot and floor and sigma and tau_s and tau_s > 0):
        return None
    st = sigma * math.sqrt(tau_s)
    p = norm.cdf(math.log(spot / floor) / st)
    if cap:
        p -= norm.cdf(math.log(spot / cap) / st)
    return p


def main():
    cutover = sys.argv[1] if len(sys.argv) > 1 else "2026-07-25T16:57:00Z"
    fills, spots = load_maker("prod")
    if len(spots) < 5:
        print("not enough spot history yet")
        return
    k = kc.Kalshi("prod")

    rows = []
    for d in fills:
        raw = d["msg"]["raw"]
        ct = raw.get("created_time") or ""
        if ct < cutover:
            continue  # 07-22 manual test fills, not this session
        t0 = d["t_local"]  # maker clock at detection (<=10s after real fill)
        floor, cap, close = market_meta(k, raw.get("ticker", ""))
        if close is None:
            continue
        close_us = datetime.strptime(close, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc).timestamp() * 1e6
        tau0 = (close_us - t0) / 1e6
        sig = d["msg"]["ctx"].get("sigma") or 3e-4
        s0 = spot_at(spots, t0)
        m = {}
        for horizon, dt in (("30s", 30e6), ("2min", 120e6)):
            s1 = spot_at(spots, t0 + dt)
            tau1 = tau0 - dt / 1e6
            f0 = fv(s0, floor, cap, sig, tau0)
            f1 = fv(s1, floor, cap, sig, tau1)
            if f0 is None or f1 is None:
                m[horizon] = None
                continue
            # our position sign: long-yes if (buy,yes) or (sell,no)
            act, side = raw.get("action"), raw.get("side")
            long_yes = (act == "buy" and side == "yes") or \
                       (act == "sell" and side == "no")
            # markout>0 = fv moved in our favor after the fill (benign);
            # <0 = moved against us (adverse selection / pickoff)
            m[horizon] = (f1 - f0) if long_yes else (f0 - f1)
        cnt = float(raw.get("count_fp") or 0)
        m["count"] = cnt
        m["ticker"] = raw.get("ticker")
        rows.append(m)

    n = len(rows)
    print(f"session fills analyzed: {n} "
          f"(spot series {len(spots)} pts, {len(fills)} total fills)")
    if n == 0:
        return
    for horizon in ("30s", "2min"):
        vals = np.array([r[horizon] for r in rows if r[horizon] is not None])
        if len(vals) == 0:
            print(f"  {horizon}: no coverage yet (need spot {horizon} past fills)")
            continue
        cents = vals * 100
        # contract-weighted too
        w = np.array([r["count"] for r in rows if r[horizon] is not None])
        wm = float(np.sum(cents * w) / np.sum(w)) if w.sum() else float("nan")
        # crude bootstrap CI over fills (NOT independent — see caveat)
        boot = [np.mean(np.random.choice(cents, len(cents))) for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  markout @{horizon}: mean {cents.mean():+.2f}c  "
              f"median {np.median(cents):+.2f}c  "
              f"contract-wtd {wm:+.2f}c  n={len(vals)}  "
              f"naive95%[{lo:+.2f},{hi:+.2f}]c")
    print("\nCAVEAT: n is tiny and fills are NOT independent (clustered on "
          "one BTC path); ctx.spot lags the true fill <=10s; the CI is naive "
          "(over fills, not BTC episodes). This is a plumbing check, not a "
          "toxicity verdict — that needs >=100 fills over >=5 BTC episodes.")


if __name__ == "__main__":
    main()
