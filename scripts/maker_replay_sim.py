"""Replay simulator: FV-maker on Polymarket BTC updown-5m, judged on
recorded tick data. The revenue model's judge — measured fills, not Fermi.

Policy per window (start s, end e = s+300s):
  quote from s+10s to e-60s, both sides of FV at safe width
  FV_t = Phi( ln(S_{t-L}/S_open) / (sigma * sqrt(e-t)) )      [lagged spot]
  delta_t = PHI0 / (sigma_1s * sqrt(e-t)) * q99|move over L|  [safe width]
  no quotes when FV outside [0.05, 0.95]; post-only vs recorded book;
  size = $100 notional per side, one clip per side at a time.

Fill rule (conservative, we are last in queue): a recorded taker BUY at
px > our_ask fills our ask; a taker SELL at px < our_bid fills our bid.
Positions hold to settlement (Binance S_e vs S_open decides Up).

Latency scenarios L in {50ms, 200ms, 500ms} share one pass.

Usage: .venv/bin/python scripts/maker_replay_sim.py
"""

from __future__ import annotations

import gzip
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket"
PHI0 = 0.3989
SIZE_USD = 100.0
LATENCIES = {"L50ms": 0.05, "L200ms": 0.2, "L500ms": 0.5}
EWMA_HL_S = 300.0


def load_lines(prefix: str):
    # gz_recover: the plain reader silently dropped every member after a
    # kill-corrupted one, not just the truncated tail
    from gz_recover import iter_jsonl
    for f in sorted(RAW.glob(f"{prefix}_*.jsonl.gz")):
        yield from iter_jsonl(f)


def token_windows() -> dict[str, int]:
    """token -> 5m window start (unix s), from discovery slugs."""
    out = {}
    for d in load_lines("poly"):
        if d["src"] == "discovery":
            for tid, meta in d["msg"].items():
                slug = meta.get("slug", "")
                if "btc-updown-5m-" in slug:
                    out[tid] = int(slug.rsplit("-", 1)[-1])
    return out


def binance_grid():
    px = [(d["t_local"], float(d["msg"]["data"]["p"]))
          for d in load_lines("binance")
          if d["src"] == "binance" and d["msg"].get("data", {}).get("s") == "BTCUSDT"]
    b = pd.DataFrame(px, columns=["t", "px"]).sort_values("t")
    b["g"] = b.t // 100_000
    g = b.groupby("g").px.last()
    grid = np.arange(g.index.min(), g.index.max() + 1)
    p = g.reindex(grid).ffill()
    lr = np.log(p).diff().fillna(0.0)
    # EWMA variance of 100ms returns -> sigma per 1s
    alpha = 1 - 0.5 ** (0.1 / EWMA_HL_S)
    var100 = lr.pow(2).ewm(alpha=alpha).mean()
    sigma_1s = np.sqrt(var100 * 10.0)
    q99 = {name: float(np.abs(np.log(p).diff(max(1, int(L * 10)))).quantile(0.99))
           for name, L in LATENCIES.items()}
    return grid, p.values, sigma_1s.values, q99


def market_events():
    """Per token: trades list and (for settlement sanity) last trade."""
    trades = defaultdict(list)
    for d in load_lines("poly"):
        if d["src"] != "poly":
            continue
        for m in (d["msg"] if isinstance(d["msg"], list) else [d["msg"]]):
            if m.get("event_type") == "last_trade_price":
                trades[m.get("asset_id")].append(
                    (d["t_local"], float(m["price"]), float(m["size"]),
                     m.get("side")))
    return trades


def main() -> None:
    toks = token_windows()
    print(f"{len(toks)} updown-5m tokens")
    grid, spot, sig1s, q99 = binance_grid()
    g0 = grid[0]

    def spot_at(t_us: int) -> float:
        i = int(t_us // 100_000 - g0)
        return spot[min(max(i, 0), len(spot) - 1)]

    def sig_at(t_us: int) -> float:
        i = int(t_us // 100_000 - g0)
        return max(sig1s[min(max(i, 0), len(sig1s) - 1)], 1e-6)

    trades = market_events()

    # group tokens by window; keep the token that正 correlates with spot (Up)
    by_win = defaultdict(list)
    for tid, ws in toks.items():
        by_win[ws].append(tid)

    results = {n: {"pnl": 0.0, "fills": 0, "pickoff": 0.0, "wins": 0}
               for n in LATENCIES}
    n_windows = 0
    for ws, tids in sorted(by_win.items()):
        s_us, e_us = ws * 1_000_000, (ws + 300) * 1_000_000
        if s_us < grid[0] * 100_000 or e_us > grid[-1] * 100_000:
            continue
        s_open, s_end = spot_at(s_us), spot_at(e_us)
        up_won = s_end >= s_open
        # pick the Up token: corr of trade prices with spot over window
        up_tid = None
        best_c = 0.0
        for tid in tids:
            tr = [x for x in trades.get(tid, []) if s_us <= x[0] <= e_us]
            if len(tr) < 8:
                continue
            px = np.array([x[1] for x in tr])
            sp = np.array([spot_at(x[0]) for x in tr])
            if px.std() < 1e-9 or sp.std() < 1e-9:
                continue
            c = float(np.corrcoef(px, sp)[0, 1])
            if abs(c) > abs(best_c):
                best_c, up_tid = c, tid if c > 0 else None
                if c < 0 and len(tids) == 2:
                    up_tid = [t for t in tids if t != tid][0]
        if up_tid is None or up_tid not in trades:
            continue
        n_windows += 1
        wtr = [x for x in trades[up_tid] if s_us <= x[0] <= e_us]

        for name, L in LATENCIES.items():
            L_us = int(L * 1e6)
            res = results[name]
            last_fill_t = {"bid": 0, "ask": 0}
            for t_tr, px, sz, side in wtr:
                if not (s_us + 10_000_000 <= t_tr <= e_us - 60_000_000):
                    continue
                tau = (e_us - t_tr) / 1e6
                sg = sig_at(t_tr)
                st = sg * np.sqrt(tau)
                fv = float(norm.cdf(np.log(spot_at(t_tr - L_us) / s_open) / st))
                if not 0.05 < fv < 0.95:
                    continue
                delta = PHI0 / st * q99[name]
                bid, ask = fv - delta, fv + delta
                if side == "BUY" and px > ask and 0 < ask < 1 \
                        and t_tr - last_fill_t["ask"] > 5e6:
                    n_ct = SIZE_USD  # $100 of contracts ~ SIZE/1 payoff units
                    pnl = (ask - (1.0 if up_won else 0.0)) * n_ct
                    res["pnl"] += pnl
                    res["fills"] += 1
                    last_fill_t["ask"] = t_tr
                    fv30 = float(norm.cdf(
                        np.log(spot_at(t_tr + 30_000_000) / s_open)
                        / (sg * np.sqrt(max(tau - 30, 30)))))
                    if fv30 - ask > 0.02:
                        res["pickoff"] += pnl
                elif side == "SELL" and px < bid and 0 < bid < 1 \
                        and t_tr - last_fill_t["bid"] > 5e6:
                    n_ct = SIZE_USD
                    pnl = ((1.0 if up_won else 0.0) - bid) * n_ct
                    res["pnl"] += pnl
                    res["fills"] += 1
                    last_fill_t["bid"] = t_tr
                    fv30 = float(norm.cdf(
                        np.log(spot_at(t_tr + 30_000_000) / s_open)
                        / (sg * np.sqrt(max(tau - 30, 30)))))
                    if bid - fv30 > 0.02:
                        res["pickoff"] += pnl
                if res["pnl"] > 0:
                    res["wins"] += 0  # placeholder, wins computed post hoc

    hours = (grid[-1] - grid[0]) / 36000
    print(f"replayed {n_windows} windows over {hours:.1f}h\n")
    print(f"{'scenario':>8s} {'fills':>6s} {'gross_pnl':>10s} "
          f"{'pickoff_part':>12s} {'pnl/day':>9s} {'per-fill':>9s}")
    for name, r in results.items():
        per_day = r["pnl"] / hours * 24
        per_fill = r["pnl"] / r["fills"] if r["fills"] else float("nan")
        print(f"{name:>8s} {r['fills']:6d} ${r['pnl']:9.2f} "
              f"${r['pickoff']:11.2f} ${per_day:8.2f} ${per_fill:8.3f}")
    print("\nnotes: $100/side clips, >=5s between fills per side, hold to "
          "settlement, fees 0 (measured), conservative last-in-queue fills; "
          "settlement via Binance proxy (Chainlink basis unmodeled)")


if __name__ == "__main__":
    main()
