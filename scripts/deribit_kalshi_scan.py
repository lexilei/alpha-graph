"""Deribit option-implied distribution vs Kalshi BTC ladder — deviation scan.

For the target Kalshi settlement (default: today's KXBTCD daily, 17:00 ET),
build P(S_T > K) from the nearest-after Deribit expiry smile and compare
with Kalshi's quoted yes bid/ask per strike.

Digital from smile with skew term:  P(S>K) = N(d2) - vega(K) * dsigma/dK
Maturity mismatch handled by evaluating the same smile at the target tau
(flat-forward-vol assumption — printed as a caveat; intraday vol seasonality
is unmodeled). Kalshi taker fee 0.07*p*(1-p) applied to tradability flags.

Usage: .venv/bin/python scripts/deribit_kalshi_scan.py
"""

from __future__ import annotations

import gzip
import json
import re
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def last_line(pattern: str, src: str, base: Path) -> dict | None:
    out = None
    for f in sorted(base.glob(pattern)):
        try:
            for line in gzip.open(f, "rt"):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d["src"] == src:
                    out = d
        except (EOFError, zlib.error):
            continue
    return out


def parse_deribit_name(name: str):
    m = re.match(r"BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])", name)
    if not m:
        return None
    day, mon, yy, k, cp = m.groups()
    exp = datetime(2000 + int(yy), MONTHS[mon], int(day), 8, 0,
                   tzinfo=timezone.utc)
    return exp, float(k), cp


def parse_kalshi_ticker(t: str):
    # KXBTCD-26JUL2217-T75749.99 / KXBTC-26JUL2217-B75250
    m = re.match(r"(KXBTCD?|KXBTC)-(\d{2})([A-Z]{3})(\d{2})(\d{2})-([TB])(.+)", t)
    if not m:
        return None
    series, yy, mon, dd, hh, kind, val = m.groups()
    # settle hour is ET; EDT in July = UTC-4
    close = datetime(2000 + int(yy), MONTHS[mon], int(dd), int(hh),
                     tzinfo=timezone.utc) + pd.Timedelta(hours=4)
    return series, close, kind, float(val)


def _kget(path: str):
    import urllib.request
    req = urllib.request.Request(
        f"https://api.elections.kalshi.com/trade-api/v2{path}",
        headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def kalshi_listing(spot: float) -> list[dict]:
    """Live listing; strikes within 8% of spot get REAL top-of-book quotes
    (the /markets bid/ask fields are unreliable on one-sided books)."""
    out = []
    for s in ("KXBTC", "KXBTCD"):
        for mk in _kget(f"/markets?series_ticker={s}&status=open&limit=500"
                        ).get("markets", []):
            row = {"ticker": mk["ticker"],
                   "floor_strike": mk.get("floor_strike"),
                   "cap_strike": mk.get("cap_strike"),
                   "yes_bid": 1.0 - float(mk.get("no_ask_dollars") or 1),
                   "yes_ask": 1.0 - float(mk.get("no_bid_dollars") or 0)}
            ks = [k for k in (row["floor_strike"], row["cap_strike"]) if k]
            if ks and any(abs(k / spot - 1) < 0.08 for k in ks):
                ob = _kget(f"/markets/{mk['ticker']}/orderbook?depth=1"
                           ).get("orderbook_fp", {})
                yb = ob.get("yes_dollars") or []
                nb = ob.get("no_dollars") or []
                row["yes_bid"] = float(yb[-1][0]) if yb else 0.0
                row["yes_ask"] = 1.0 - float(nb[-1][0]) if nb else 1.0
            out.append(row)
    return out


def main() -> None:
    der = last_line("deribit_*.jsonl.gz", "deribit_surface", RAW / "deribit")
    now = datetime.now(timezone.utc)
    spot = der["msg"]["index"]["index_price"]
    kal = {"msg": kalshi_listing(spot)}
    n_mkts = len(kal["msg"])
    print(f"deribit snapshot age: {(now.timestamp() - der['t_local']/1e6)/60:.1f}min"
          f" | index ${spot:,.0f} | kalshi live listing: {n_mkts} markets")

    # all future settlements 1h..36h out, each judged on its own ladder
    rows = []
    for mkt in kal["msg"]:
        p = parse_kalshi_ticker(mkt["ticker"])
        if not p:
            continue
        series, close, kind, val = p
        h = (close - now).total_seconds() / 3600
        if 1.0 <= h <= 36.0:
            rows.append((close, mkt, kind, val))
    if not rows:
        print("no ladders 1-36h out in the last listing")
        return

    all_opts = []
    for o in der["msg"]["options"]:
        pr = parse_deribit_name(o["instrument_name"])
        if pr and o.get("mark_iv"):
            all_opts.append((*pr, o["mark_iv"] / 100.0))

    n_sig_total = 0
    for target in sorted({r[0] for r in rows}):
        ladder = [(m, kind, val) for c, m, kind, val in rows if c == target]
        tau_t = (target - now).total_seconds() / (365 * 86400)
        after = [o for o in all_opts if o[0] > target]
        if not after:
            continue
        exp0 = min(o[0] for o in after)
        smile = pd.DataFrame(
            [(k, iv) for e, k, cp, iv in after if e == exp0 and cp == "C"],
            columns=["k", "iv"]).groupby("k").iv.mean().sort_index()
        logm = np.log(smile.index.values / spot)
        iv = smile.values
        dk = np.gradient(iv, smile.index.values)

        def p_above(K):
            s = float(np.interp(np.log(K / spot), logm, iv))
            st = s * np.sqrt(tau_t)
            d1 = (np.log(spot / K) + 0.5 * st * st) / st
            vega = spot * norm.pdf(d1) * np.sqrt(tau_t)
            dsg = float(np.interp(np.log(K / spot), logm, dk))
            return float(np.clip(norm.cdf(d1 - st) - vega * dsg, 0, 1))

        out = []
        for mkt, kind, val in ladder:
            if kind == "T":
                p_fair, label = p_above(val), f"T>{val:,.0f}"
            else:
                fs, cs = mkt.get("floor_strike"), mkt.get("cap_strike")
                if fs is None or cs is None:
                    continue
                p_fair = p_above(fs) - p_above(cs)
                label = f"B{fs:,.0f}-{cs:,.0f}"
            bid, ask = mkt["yes_bid"], mkt["yes_ask"]
            mid = (bid + ask) / 2
            fee = 0.07 * mid * (1 - mid)
            dev = p_fair - mid
            trade = ("BUY" if p_fair > ask + fee else
                     "SELL" if p_fair < bid - fee else "")
            out.append({"market": label, "bid": bid, "ask": ask,
                        "deribit_p": round(p_fair, 3),
                        "dev": round(dev, 3), "fee": round(fee, 3),
                        "sig": trade})
        df = pd.DataFrame(out).sort_values("dev", key=abs, ascending=False)
        n_sig = (df.sig != "").sum()
        n_sig_total += n_sig
        print(f"\n== settle {target} (tau {tau_t*365*24:.1f}h, deribit exp "
              f"{exp0.strftime('%d%b %H:%M')}): {len(df)} strikes, "
              f"{n_sig} cross fee band ==")
        with pd.option_context("display.width", 160):
            print(df.head(10).to_string(index=False))
    print(f"\nTOTAL fee-band crossings: {n_sig_total} "
          f"(taker fee; maker halves the bar)")
    print("caveats: flat-forward-vol maturity scaling; deribit RND is "
          "risk-neutral (VRP wedge); kalshi listing bid/ask may be stale "
          "up to 60s; single snapshot, not a judgment")


if __name__ == "__main__":
    main()
