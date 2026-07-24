"""Safe-width curve + maker revenue model for Polymarket BTC updown markets.

Inputs are all empirical, from the recorded streams:
  - Binance BTC 1s vol and |move| tail quantiles over cancel-latency windows
  - Polymarket trade prints (last_trade_price: price, size, side, fee_rate_bps)
  - Book snapshots for prevailing mid at trade time and resting spreads

Outputs:
  1. fee_rate_bps distribution across actual trades (fee-incidence evidence)
  2. safe half-width delta(tau; L): ATM binary sensitivity phi(0)/(sigma*sqrt(tau))
     times the empirical q99 |move| over cancel latency L
  3. observed flow: notional/hour on 5m tokens, trade distance-from-mid
     distribution -> captured-flow fraction at each width
  4. first-order $/day revenue for a maker at width delta(L) capturing a
     share of that flow, with a q99-miss pickoff penalty

Usage: .venv/bin/python scripts/maker_width_analysis.py
"""

from __future__ import annotations

import gzip

from gz_recover import iter_jsonl
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket"
PHI0 = 0.3989
LATENCIES_S = {"vps_50ms": 0.05, "home_200ms": 0.2, "mac_500ms": 0.5}
REBATE = 0.0035  # 35bp of filled notional if fee incidence confirms; shown separately


def load_lines(prefix: str):
    # gz_recover: plain gzip readers crash on kill-corrupted members and
    # silently drop everything after them (nightly batch died on this)
    for f in sorted(RAW.glob(f"{prefix}_*.jsonl.gz")):
        yield from iter_jsonl(f)


def btc5m_tokens() -> set[str]:
    toks = set()
    for d in load_lines("poly"):
        if d["src"] == "discovery":
            for tid, meta in d["msg"].items():
                if "btc-updown-5m" in meta.get("slug", ""):
                    toks.add(tid)
    return toks


def main() -> None:
    toks = btc5m_tokens()

    # ---- pass over poly stream: trades + mids from book snapshots ----
    fee_counter: Counter = Counter()
    trades = []          # (t_local, token, price, size, side)
    mids: dict[str, list[tuple[int, float]]] = defaultdict(list)
    spreads = []
    for d in load_lines("poly"):
        if d["src"] != "poly":
            continue
        msgs = d["msg"] if isinstance(d["msg"], list) else [d["msg"]]
        for m in msgs:
            et = m.get("event_type")
            aid = m.get("asset_id")
            if et == "last_trade_price" and aid in toks:
                fee_counter[m.get("fee_rate_bps")] += 1
                trades.append((d["t_local"], aid, float(m["price"]),
                               float(m["size"]), m.get("side")))
            elif et == "book" and aid in toks:
                bids = [float(x["price"]) for x in m.get("bids", [])]
                asks = [float(x["price"]) for x in m.get("asks", [])]
                if bids and asks:
                    bb, ba = max(bids), min(asks)
                    if 0 < bb <= ba < 1:
                        mids[aid].append((d["t_local"], (bb + ba) / 2))
                        if 0.15 < (bb + ba) / 2 < 0.85:
                            spreads.append(ba - bb)
    tr = pd.DataFrame(trades, columns=["t", "token", "price", "size", "side"])
    tr["notional"] = tr.price * tr["size"]
    hours = (tr.t.max() - tr.t.min()) / 3.6e9
    print(f"5m-token trades: {len(tr):,} | recorded span {hours:.1f}h | "
          f"taker notional ${tr.notional.sum():,.0f} "
          f"(${tr.notional.sum()/hours:,.0f}/hour overnight)")
    print(f"fee_rate_bps on trades: {dict(fee_counter)}")
    sp = pd.Series(spreads)
    print(f"resting spread (mid 15-85c): p50 {sp.quantile(0.5)*100:.1f}c "
          f"p90 {sp.quantile(0.9)*100:.1f}c")

    # trade distance from prevailing mid = effective half-spread paid by taker
    dist = []
    for tok, g in tr.groupby("token"):
        ms = mids.get(tok)
        if not ms:
            continue
        mt = np.array([x[0] for x in ms])
        mv = np.array([x[1] for x in ms])
        idx = np.searchsorted(mt, g.t.values) - 1
        ok = idx >= 0
        d_ = np.abs(g.price.values[ok] - mv[idx[ok]])
        dist.append(pd.DataFrame({"d": d_, "notional": g.notional.values[ok]}))
    dist = pd.concat(dist)
    print("\ntaker fill distance from mid (notional-weighted share of flow "
          "at distance >= x):")
    for x in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        share = dist.notional[dist.d >= x].sum() / dist.notional.sum()
        print(f"  >= {x*100:3.1f}c : {share:5.1%}")

    # ---- Binance vol + move tails ----
    px = []
    for d in load_lines("binance"):
        if d["src"] == "binance":
            m = d["msg"].get("data", {})
            if m.get("s") == "BTCUSDT":
                px.append((d["t_local"], float(m["p"])))
    b = pd.DataFrame(px, columns=["t", "px"]).sort_values("t")
    b["g100"] = b.t // 100_000
    g = b.groupby("g100").px.last()
    grid = np.arange(g.index.min(), g.index.max() + 1)
    p100 = g.reindex(grid).ffill()
    lr = np.log(p100).diff()
    sigma_1s = float(lr.rolling(10).sum().std())
    print(f"\nBTC 1s vol (this sample): {sigma_1s*1e4:.2f}bp")
    tails = {}
    for name, L in LATENCIES_S.items():
        k = max(1, int(L * 10))
        mv = np.log(p100).diff(k).abs().dropna()
        tails[name] = float(mv.quantile(0.99))
        print(f"  q99 |move| over {name}: {tails[name]*1e4:.2f}bp")

    # ---- safe width curve ----
    print("\nsafe ATM half-width delta(tau) = phi(0)/(sigma*sqrt(tau)) * q99(L), cents:")
    taus = [240, 180, 120, 60, 30, 15]
    header = "  tau_s " + "".join(f"{n:>12s}" for n in LATENCIES_S)
    print(header)
    curve = {}
    for tau in taus:
        sens = PHI0 / (sigma_1s * np.sqrt(tau))
        row = {n: sens * tails[n] for n in LATENCIES_S}
        curve[tau] = row
        print(f"  {tau:5d} " + "".join(f"{row[n]*100:11.1f}c" for n in LATENCIES_S))

    # ---- revenue model ----
    # assume we quote both sides at width delta(L) for the first 4 minutes of
    # each window (skip the last minute: width explodes), capture the flow
    # that trades at distance >= our width, share s of it (competition), earn
    # distance + rebate; lose on the 1% of moves beyond q99 (pickoff ~ mid
    # jump size measured yesterday: 12c median).
    daily_notional = tr.notional.sum() / hours * 24
    print(f"\nrevenue model (scaling overnight flow to 24h ~ ${daily_notional:,.0f}"
          f" 5m-token taker notional/day, x2-4 for daytime not measured):")
    jump_rate_day = 15 / 5.5 * 24  # jumps/day from yesterday's count
    for n in LATENCIES_S:
        w = np.mean([curve[t][n] for t in (240, 180, 120)])  # early-window width
        share_flow = float(dist.notional[dist.d >= w].sum() / dist.notional.sum())
        for s in (0.10, 0.25):
            captured = daily_notional * share_flow * s
            spread_rev = captured * w
            rebate_rev = captured * REBATE
            # pickoff: 1% of jumps exceed q99; assume full 12c loss on
            # rewardsMinSize-ish resting size ~ $50/side when hit
            pickoff = 0.01 * jump_rate_day * 0.12 * 50
            net = spread_rev + rebate_rev - pickoff
            print(f"  {n:>12s} w={w*100:4.1f}c flow-share>={share_flow:5.1%} "
                  f"our-share {s:.0%}: spread ${spread_rev:6.0f} + rebate "
                  f"${rebate_rev:5.0f} - pickoff ${pickoff:4.0f} = ${net:6.0f}/day")


if __name__ == "__main__":
    main()
