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
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from gz_recover import iter_jsonl

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
        for d in iter_jsonl(f):
            if d.get("src") == src:
                out = d
    return out


def parse_deribit_name(name: str):
    m = re.match(r"(BTC|ETH)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])", name)
    if not m:
        return None
    cur, day, mon, yy, k, cp = m.groups()
    exp = datetime(2000 + int(yy), MONTHS[mon], int(day), 8, 0,
                   tzinfo=timezone.utc)
    return cur, exp, float(k), cp


def parse_kalshi_ticker(t: str):
    # KXBTCD-26JUL2217-T75749.99 / KXBTC-26JUL2217-B75250
    m = re.match(r"(KXBTCD|KXBTC|KXETHD|KXETH)-(\d{2})([A-Z]{3})(\d{2})(\d{2})-([TB])(.+)", t)
    if not m:
        return None
    series, yy, mon, dd, hh, kind, val = m.groups()
    # settle hour is US/Eastern; resolve the actual UTC offset per date
    from zoneinfo import ZoneInfo
    close = datetime(2000 + int(yy), MONTHS[mon], int(dd), int(hh),
                     tzinfo=ZoneInfo("America/New_York")).astimezone(
                         timezone.utc)
    return series, close, kind, float(val)


def _kget(path: str):
    import urllib.request
    req = urllib.request.Request(
        f"https://api.elections.kalshi.com/trade-api/v2{path}",
        headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def kalshi_listing(spots: dict) -> list[dict]:
    """Live listing; strikes within 8% of the SERIES' OWN spot get real
    top-of-book quotes (listing bid/ask lies on one-sided books)."""
    out = []
    for s in ("KXBTC", "KXBTCD", "KXETH", "KXETHD"):
        spot = spots.get({"KXBTC": "BTC", "KXBTCD": "BTC",
                          "KXETH": "ETH", "KXETHD": "ETH"}[s])
        if not spot:
            continue
        for mk in _kget(f"/markets?series_ticker={s}&status=open&limit=500"
                        ).get("markets", []):
            row = {"ticker": mk["ticker"],
                   "floor_strike": mk.get("floor_strike"),
                   "cap_strike": mk.get("cap_strike"),
                   "yes_bid": 1.0 - float(mk.get("no_ask_dollars") or 1),
                   "yes_ask": 1.0 - float(mk.get("no_bid_dollars") or 0)}
            ks = [k for k in (row["floor_strike"], row["cap_strike"]) if k]
            row["has_ob"] = bool(ks) and any(
                abs(k / spot - 1) < 0.08 for k in ks)
            if row["has_ob"]:
                ob = _kget(f"/markets/{mk['ticker']}/orderbook?depth=1"
                           ).get("orderbook_fp", {})
                yb = ob.get("yes_dollars") or []
                nb = ob.get("no_dollars") or []
                row["yes_bid"] = float(yb[-1][0]) if yb else 0.0
                row["yes_ask"] = 1.0 - float(nb[-1][0]) if nb else 1.0
            out.append(row)
    return out


def brti_spot(currency: str = "BTC") -> float | None:
    """VW mid of the last ~10s of recorded Coinbase+Kraken trades for the
    given currency — the settlement-relevant anchor. Handles both the
    legacy (BTC-only) and the tagged (prod/pair) record formats."""
    want_cb = f"{currency}-USD"
    want_kr = ("XBT/USD" if currency == "BTC" else f"{currency}/USD")
    files = sorted((RAW / "polymarket").glob("brti_*.jsonl.gz"))
    if not files:
        return None
    rows = []
    for f in files[-2:]:
        for d in iter_jsonl(f):
            if d["src"] == "coinbase":
                m = d["msg"]
                prod = m.get("prod") or "BTC-USD"  # legacy rows were BTC
                if prod == want_cb:
                    rows.append((d["t_local"], float(m["p"]), float(m["s"])))
            elif d["src"] == "kraken":
                m = d["msg"]
                if isinstance(m, dict):
                    if m.get("pair") == want_kr:
                        for tr in m.get("trades", []):
                            rows.append((d["t_local"], float(tr[0]),
                                         float(tr[1])))
                elif currency == "BTC":  # legacy list rows were BTC
                    for tr in m:
                        rows.append((d["t_local"], float(tr[0]), float(tr[1])))
    if not rows:
        return None
    tmax = rows[-1][0]
    recent = [(p_, v) for t, p_, v in rows if t >= tmax - 10_000_000]
    if not recent:
        return None
    ps = [p_ for p_, v in recent]
    ws = [max(v, 1e-9) for p_, v in recent]
    return float(sum(p_ * w for p_, w in zip(ps, ws)) / sum(ws))


def last_surfaces() -> dict:
    """Last deribit_surface record per currency (records are tagged since
    the ETH extension; untagged legacy records count as BTC)."""
    out = {}
    for f in sorted((RAW / "deribit").glob("deribit_*.jsonl.gz"))[-3:]:
        for d in iter_jsonl(f):
            if d.get("src") == "deribit_surface":
                out[d["msg"].get("currency", "BTC")] = d
    return out


SERIES_CUR = {"KXBTC": "BTC", "KXBTCD": "BTC", "KXETH": "ETH", "KXETHD": "ETH"}


def main(sink=None) -> None:
    surfaces = last_surfaces()
    now = datetime.now(timezone.utc)
    spots = {}
    for cur, der in surfaces.items():
        anchor = brti_spot(cur)
        spots[cur] = anchor or der["msg"]["index"]["index_price"] * (1 - 6e-4)
    spot = spots.get("BTC")
    if spot is None:
        print("no BTC surface recorded yet")
        return
    kal = {"msg": kalshi_listing(spots)}
    n_mkts = len(kal["msg"])
    print(f"surfaces: {sorted(surfaces)} | BTC ${spot:,.0f}"
          + (f" | ETH ${spots['ETH']:,.0f}" if "ETH" in spots else "")
          + f" | kalshi live listing: {n_mkts} markets")

    # all future settlements 1h..36h out, each judged on its own ladder
    rows = []
    for mkt in kal["msg"]:
        p = parse_kalshi_ticker(mkt["ticker"])
        if not p:
            continue
        series, close, kind, val = p
        h = (close - now).total_seconds() / 3600
        if 1.0 <= h <= 36.0:
            rows.append((close, mkt, kind, val, SERIES_CUR[series]))
    if not rows:
        print("no ladders 1-36h out in the last listing")
        return

    all_opts = {}  # currency -> [(exp, k, cp, iv)]
    for cur, der in surfaces.items():
        for o in der["msg"]["options"]:
            pr = parse_deribit_name(o["instrument_name"])
            if pr and o.get("mark_iv"):
                _, exp, k, cp = pr
                all_opts.setdefault(cur, []).append(
                    (exp, k, cp, o["mark_iv"] / 100.0))

    n_sig_total = 0
    for target, lcur in sorted({(r[0], r[4]) for r in rows}):
        ladder = [(m, kind, val) for c, m, kind, val, cu in rows
                  if c == target and cu == lcur]
        cspot = spots.get(lcur)
        if cspot is None:
            continue
        spot = cspot
        tau_t = (target - now).total_seconds() / (365 * 86400)
        after = [o for o in all_opts.get(lcur, []) if o[0] > target]
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
            # signal only on strikes with a real orderbook read: the listing
            # bid/ask fields fabricate 0.00/0.01 on one-sided books
            trade = ""
            if mkt.get("has_ob"):
                trade = ("BUY" if p_fair > ask + fee else
                         "SELL" if p_fair < bid - fee else "")
            out.append({"market": label, "ticker": mkt["ticker"],
                        "bid": bid, "ask": ask,
                        "deribit_p": round(p_fair, 4),
                        "dev": round(dev, 4), "fee": round(fee, 4),
                        "tau_h": round(tau_t * 365 * 24, 2), "sig": trade})
        df = pd.DataFrame(out).sort_values("dev", key=abs, ascending=False)
        n_sig = (df.sig != "").sum()
        n_sig_total += n_sig
        if sink is not None:
            sink.write("devscan", {"settle": str(target), "cur": lcur,
                                   "spot": spot, "rows": out})
        else:
            print(f"\n== settle {target} {lcur} (tau {tau_t*365*24:.1f}h, deribit "
                  f"exp {exp0.strftime('%d%b %H:%M')}): {len(df)} strikes, "
                  f"{n_sig} cross fee band ==")
            with pd.option_context("display.width", 160):
                print(df.head(10).to_string(index=False))
    if sink is not None:
        sink.flush()
        return
    print(f"\nTOTAL fee-band crossings: {n_sig_total} "
          f"(taker fee; maker halves the bar)")
    print("caveats: flat-forward-vol maturity scaling; deribit RND is "
          "risk-neutral (VRP wedge); kalshi listing bid/ask may be stale "
          "up to 60s; single snapshot, not a judgment")


if __name__ == "__main__":
    import sys
    import time as _time
    sys.stdout.reconfigure(line_buffering=True)  # log stays live when stdout is a file
    if "--loop" in sys.argv:
        import polymarket_recorder as rec
        _sink = rec.Sink("devscan")
        period = float(sys.argv[sys.argv.index("--loop") + 1])
        while True:
            t0 = _time.time()
            try:
                main(sink=_sink)
            except Exception as e:  # noqa: BLE001
                _sink.write("devscan_err", str(e))
                _sink.flush()
            _time.sleep(max(0.0, period - (_time.time() - t0)))
    else:
        main()
