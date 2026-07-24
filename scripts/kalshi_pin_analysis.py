"""Kalshi KXBTC hourly brackets vs Binance spot: pin study on the 10s books.

For every recorded orderbook snapshot of the bracket containing Binance spot
(the pinned bracket), inside the final 10 minutes before settlement:

  p_impl   = Phi((cap-spot)/sig_rem) - Phi((floor-spot)/sig_rem)
             terminal-normal implied hit prob, sig_rem = trailing 10-min
             realized vol scaled to the remaining horizon
  buy_edge  = p_impl - yes_ask - fee(yes_ask)
  sell_edge = yes_bid - p_impl - fee(yes_bid)

fee = Kalshi taker 0.07*p*(1-p). Quotes come from the 10s orderbook stream
(yes_bid = best yes_dollars, yes_ask = 1 - best no_dollars), not the 60s
market lists, which are stale exactly when it matters.

Caveats pinned upfront: settlement proxied by the last Binance trade before
close (Kalshi settles on its own index — a few $ of slack, occasionally
decisive); terminal-normal understates fat tails, so real hit probs are
below p_impl near the edges — shade buy_edge down.

Usage: .venv/bin/python scripts/kalshi_pin_analysis.py
"""

from __future__ import annotations

import gzip
import json
import zlib
from datetime import datetime, timezone
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket"
WINDOW_US = 600 * 1_000_000  # study the final 10 minutes
VOL_US = 600 * 1_000_000     # trailing window for realized vol


def load(prefix: str):
    # gz_recover: the plain reader silently dropped every member after a
    # kill-corrupted one, not just the truncated tail
    from gz_recover import iter_jsonl
    for f in sorted(RAW.glob(f"{prefix}_*.jsonl.gz")):
        yield from iter_jsonl(f)


def phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def kalshi_fee(p: float) -> float:
    return 0.07 * p * (1.0 - p)


def load_kalshi():
    meta: dict[str, tuple[int, float, float]] = {}
    books = []
    for d in load("kalshi"):
        if d["src"] == "kalshi_markets":
            for mk in d["msg"]:
                if mk.get("floor_strike") is None or mk.get("cap_strike") is None:
                    continue
                close_us = int(datetime.strptime(
                    mk["close_time"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc).timestamp() * 1e6)
                meta[mk["ticker"]] = (close_us, float(mk["floor_strike"]),
                                      float(mk["cap_strike"]))
        elif d["src"] == "kalshi":
            ob = d["msg"]["ob"].get("orderbook_fp") or {}
            yes = ob.get("yes_dollars") or []
            no = ob.get("no_dollars") or []
            if not yes or not no:
                continue
            yb, yb_sz = max(((float(p), float(s)) for p, s in yes))
            nb, nb_sz = max(((float(p), float(s)) for p, s in no))
            books.append((d["t_local"], d["msg"]["ticker"],
                          yb, 1.0 - nb, yb_sz, nb_sz))
    bdf = pd.DataFrame(books, columns=[
        "t", "ticker", "yes_bid", "yes_ask", "bid_sz", "ask_sz"])
    return meta, bdf


def btc_spot() -> pd.DataFrame:
    rows = []
    for d in load("binance"):
        m = d["msg"].get("data", {})
        if m.get("s") == "BTCUSDT":
            rows.append((d["t_local"], float(m["p"])))
    return pd.DataFrame(rows, columns=["t", "px"]).sort_values("t")


def main() -> None:
    meta, books = load_kalshi()
    spot = btc_spot()
    print(f"tickers: {len(meta)} | book snapshots: {len(books):,} | "
          f"btc trades: {len(spot):,}")

    st, sp = spot.t.values, spot.px.values
    # per-second grid for realized vol
    sec = pd.Series(sp, index=st // 1_000_000).groupby(level=0).last()
    grid = np.arange(sec.index.min(), sec.index.max() + 1)
    px1s = sec.reindex(grid).ffill().values
    lr2 = np.square(np.diff(np.log(px1s), prepend=np.log(px1s[0])))
    csum = np.cumsum(lr2)

    rows = []
    for r in books.itertuples():
        m = meta.get(r.ticker)
        if m is None:
            continue
        close_us, floor, cap = m
        ttc = close_us - r.t
        if not 0 < ttc <= WINDOW_US:
            continue
        i_now = np.searchsorted(st, r.t) - 1
        if i_now < 1:
            continue
        px = sp[i_now]
        if not floor <= px <= cap:
            continue
        g_now = int(r.t // 1_000_000 - grid[0])
        g_v0 = g_now - 600
        if g_v0 < 1 or g_now >= len(csum):
            continue
        sig10 = sqrt(max(csum[g_now] - csum[g_v0], 1e-12)) * px
        sig_rem = sig10 * sqrt(ttc / VOL_US)
        p_impl = phi((cap - px) / sig_rem) - phi((floor - px) / sig_rem)
        i_set = np.searchsorted(st, close_us) - 1
        settle = sp[i_set] if i_set < len(sp) else np.nan
        rows.append({
            "close": datetime.fromtimestamp(
                close_us / 1e6, tz=timezone.utc).strftime("%m-%d %H:%M"),
            "ttc_s": ttc / 1e6, "spot": px, "floor": floor, "cap": cap,
            "yes_bid": r.yes_bid, "yes_ask": r.yes_ask,
            "bid_sz": r.bid_sz, "ask_sz": r.ask_sz,
            "p_impl": p_impl,
            "buy_edge": p_impl - r.yes_ask - kalshi_fee(r.yes_ask),
            "sell_edge": r.yes_bid - p_impl - kalshi_fee(r.yes_bid),
            "hit": bool(floor <= settle <= cap),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("no pinned-bracket book snapshots in the final 10 minutes")
        return
    print(f"\npinned-bracket snapshots (final 10 min): {len(df):,} "
          f"across {df.close.nunique()} settlements")

    print("\nbuy_edge (p_impl - ask - fee) distribution:")
    q = df.buy_edge.quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    print("  " + "  ".join(f"p{int(k*100)} {v:+.3f}" for k, v in q.items()))
    print("sell_edge (bid - p_impl - fee) distribution:")
    q = df.sell_edge.quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    print("  " + "  ".join(f"p{int(k*100)} {v:+.3f}" for k, v in q.items()))

    # calibration: did cheap high-p_impl asks actually pay?
    for lo, hi in ((0.90, 1.01), (0.75, 0.90), (0.5, 0.75)):
        g = df[(df.p_impl >= lo) & (df.p_impl < hi)]
        if len(g) < 5:
            continue
        print(f"p_impl [{lo:.2f},{hi:.2f}): n={len(g):4d}  "
              f"hit {g.hit.mean():.0%}  median ask {g.yes_ask.median():.2f}  "
              f"median buy_edge {g.buy_edge.median():+.3f}")

    big = df[df.buy_edge > 0.05].sort_values("buy_edge", ascending=False)
    print(f"\nsnapshots with buy_edge > 5c: {len(big)} "
          f"({len(big)/len(df):.1%}) | median ask size there: "
          f"{big.ask_sz.median() if len(big) else float('nan'):.0f}")
    if len(big):
        agg = (big.groupby("close")
               .agg(n=("buy_edge", "size"), max_edge=("buy_edge", "max"),
                    hit=("hit", "mean")))
        print(agg.to_string())


if __name__ == "__main__":
    main()
