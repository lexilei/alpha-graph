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


def main() -> None:
    der = last_line("deribit_*.jsonl.gz", "deribit_surface", RAW / "deribit")
    kal = last_line("kalshi_*.jsonl.gz", "kalshi_markets", RAW / "polymarket")
    now = datetime.now(timezone.utc)
    spot = der["msg"]["index"]["price"]
    print(f"deribit snapshot age: {(now.timestamp() - der['t_local']/1e6)/60:.1f}min"
          f" | index ${spot:,.0f} | kalshi list age: "
          f"{(now.timestamp() - kal['t_local']/1e6)/60:.1f}min")

    # target = the KXBTCD close nearest in the future
    rows = []
    for mkt in kal["msg"]:
        p = parse_kalshi_ticker(mkt["ticker"])
        if p and p[0] == "KXBTCD" and p[1] > now:
            rows.append((p[1], mkt, p[2], p[3]))
    if not rows:
        print("no future KXBTCD markets in the last listing")
        return
    target = min(r[0] for r in rows)
    ladder = [(m, kind, val) for c, m, kind, val in rows if c == target]
    tau_t = (target - now).total_seconds() / (365 * 86400)
    print(f"target settlement: {target} (tau = {tau_t*365*24:.1f}h), "
          f"{len(ladder)} markets on ladder")

    # nearest-after deribit expiry smile
    opts = []
    for o in der["msg"]["options"]:
        pr = parse_deribit_name(o["instrument_name"])
        if not pr or not o.get("mark_iv"):
            continue
        exp, k, cp = pr
        if exp > target:
            opts.append((exp, k, cp, o["mark_iv"] / 100.0))
    exp0 = min(o[0] for o in opts)
    smile = pd.DataFrame([(k, iv) for e, k, cp, iv in opts
                          if e == exp0 and cp == "C"],
                         columns=["k", "iv"]).groupby("k").iv.mean().sort_index()
    print(f"using deribit expiry {exp0} ({len(smile)} strikes), "
          f"flat-forward-vol scaling to target tau")

    logm = np.log(smile.index.values / spot)
    iv = smile.values
    def sigma_at(K):
        return float(np.interp(np.log(K / spot), logm, iv))
    dk = np.gradient(iv, smile.index.values)
    def dsigma_at(K):
        return float(np.interp(np.log(K / spot), logm, dk))

    def p_above(K):
        s = sigma_at(K)
        st = s * np.sqrt(tau_t)
        d1 = (np.log(spot / K) + 0.5 * st * st) / st
        d2 = d1 - st
        vega = spot * norm.pdf(d1) * np.sqrt(tau_t)
        return float(np.clip(norm.cdf(d2) - vega * dsigma_at(K) / 1.0, 0, 1))

    out = []
    for mkt, kind, val in ladder:
        if kind == "T":
            p_fair = p_above(val)
            label = f"T>{val:,.0f}"
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
        out.append({"market": label, "kalshi_bid": bid, "kalshi_ask": ask,
                    "deribit_p": round(p_fair, 3), "dev_vs_mid": round(dev, 3),
                    "fee": round(fee, 3), "signal": trade})
    df = pd.DataFrame(out).sort_values("dev_vs_mid", key=abs, ascending=False)
    with pd.option_context("display.width", 160):
        print(df.to_string(index=False))
    n_sig = (df.signal != "").sum()
    print(f"\n{n_sig} strikes cross the fee-adjusted band "
          f"(taker fee; maker halves the bar)")
    print("caveats: flat-forward-vol maturity scaling; deribit RND is "
          "risk-neutral (VRP wedge); kalshi listing bid/ask may be stale "
          "up to 60s; single snapshot, not a judgment")


if __name__ == "__main__":
    main()
