"""Sportsbook consensus vs Polymarket MLB moneylines: edge + true-arb scan.

Reads the odds_poller output. For every book_odds snapshot, devig each
bookmaker's h2h prices (proportional), build a consensus fair prob per team
(Pinnacle if present, else the cross-book mean), match the game to the
Polymarket moneyline (team-name pair from the gamma question/outcomes), and
compute:

  edge_buy   = p_fair - bestAsk          (EV of buying that team on Poly)
  arb_cost   = ask_poly(A) + 1/odds(B)   (< 1 = cross-venue lock, pre-fee)

Reports the distribution over snapshots and the best cases, with the last
snapshot as the "now" table. No positions, measurement only.

Usage: .venv/bin/python scripts/odds_edge_analysis.py
"""

from __future__ import annotations

import gzip
import json
import zlib
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "odds"
POLY_TAKER_FEE = 0.01  # since 2026 Q1


def load(src_filter: set[str]):
    # gz_recover: the plain reader silently dropped every member after a
    # kill-corrupted one, not just the truncated tail
    from gz_recover import iter_jsonl
    for f in sorted(RAW.glob("odds_*.jsonl.gz")):
        for d in iter_jsonl(f):
            if d["src"] in src_filter:
                yield d


def devig(outcomes: list[dict]) -> dict[str, float] | None:
    try:
        inv = {o["name"]: 1.0 / float(o["price"]) for o in outcomes}
    except (KeyError, ZeroDivisionError, TypeError):
        return None
    s = sum(inv.values())
    if not 0.9 < s < 1.3:
        return None
    return {k: v / s for k, v in inv.items()}


def snapshot_rows(d: dict) -> list[dict]:
    rows = []
    for ev in d["msg"].get("events", []):
        teams = frozenset([ev["home_team"], ev["away_team"]])
        per_book, best_odds = {}, {}
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] != "h2h":
                    continue
                dv = devig(m["outcomes"])
                if dv:
                    per_book[bk["key"]] = dv
                for o in m["outcomes"]:
                    p = float(o["price"])
                    if p > best_odds.get(o["name"], 0.0):
                        best_odds[o["name"]] = p
        if not per_book:
            continue
        fair = per_book.get("pinnacle") or {
            t: pd.Series([b[t] for b in per_book.values() if t in b]).mean()
            for t in teams}
        # ET game date to disambiguate series games (same team pair on
        # consecutive days); commence is UTC, EDT = UTC-4
        cm = ev.get("commence_time") or ""
        try:
            cm_us = pd.Timestamp(cm).value // 1000
            date_et = (pd.Timestamp(cm) - pd.Timedelta(hours=4)).strftime(
                "%Y-%m-%d")
        except ValueError:
            continue
        rows.append({"t": d["t_local"], "teams": teams,
                     "commence_us": cm_us, "date_et": date_et,
                     "fair": fair, "best_odds": best_odds,
                     "n_books": len(per_book)})
    return rows


def main() -> None:
    odds_snaps = [snapshot_rows(d) for d in load({"book_odds"})]
    poly_snaps = [d for d in load({"poly_games"})]
    print(f"odds snapshots: {len(odds_snaps)} | poly snapshots: {len(poly_snaps)}")

    obs = []
    for d in poly_snaps:
        t = d["t_local"]
        # nearest odds snapshot at or before t (fall back to nearest after)
        cands = [s for s in odds_snaps if s and abs(s[0]["t"] - t) < 20 * 60e6]
        if not cands:
            continue
        osnap = min(cands, key=lambda s: abs(s[0]["t"] - t))
        by_key = {(r["teams"], r["date_et"]): r for r in osnap}
        for g in d["msg"]:
            try:
                outs = json.loads(g["outcomes"])
            except (TypeError, json.JSONDecodeError):
                continue
            r = by_key.get((frozenset(outs), g["slug"][-10:]))
            if r is None or g.get("bestAsk") is None or g.get("bestBid") is None:
                continue
            if "vs." not in (g.get("question") or ""):
                continue  # moneyline only, skip props/spreads
            if r["commence_us"] <= t:
                continue  # pre-game only: in-play books devig differently
            ask_a, bid_a = float(g["bestAsk"]), float(g["bestBid"])
            a, b = outs[0], outs[1]
            if a not in r["fair"] or b not in r["fair"]:
                continue
            for team, ask, bid in ((a, ask_a, bid_a),
                                   (b, 1 - bid_a, 1 - ask_a)):
                other = b if team == a else a
                obs.append({
                    "t": t, "slug": g["slug"], "team": team,
                    "fair": r["fair"][team], "ask": ask, "bid": bid,
                    "edge_buy": r["fair"][team] - ask - POLY_TAKER_FEE,
                    "arb_cost": ask + 1.0 / r["best_odds"].get(other, 1e-9),
                    "n_books": r["n_books"], "volume": float(g["volume"] or 0),
                })
    df = pd.DataFrame(obs)
    if df.empty:
        print("no matched game/odds pairs")
        return
    print(f"matched (snapshot, game, side) observations: {len(df)} "
          f"across {df.slug.nunique()} games")

    print("\n--- positive-EV buys (fair - ask - 1c fee > 0), all snapshots ---")
    pos = df[df.edge_buy > 0].sort_values("edge_buy", ascending=False)
    cols = ["slug", "team", "fair", "bid", "ask", "edge_buy", "n_books",
            "volume"]
    print(pos[cols].to_string(index=False) if len(pos) else "none")

    print("\n--- true cross-venue arbs (ask + 1/odds < 1, pre-fee) ---")
    arbs = df[df.arb_cost < 1.0].sort_values("arb_cost")
    print(arbs[["slug", "team", "ask", "arb_cost", "t"]].to_string(index=False)
          if len(arbs) else "none")

    print("\n--- last snapshot, per game ---")
    last = df[df.t == df.t.max()]
    print(last[cols].sort_values("edge_buy", ascending=False)
          .to_string(index=False))

    print("\nedge_buy distribution (all obs): "
          f"p50 {df.edge_buy.quantile(0.5):+.3f}  "
          f"p90 {df.edge_buy.quantile(0.9):+.3f}  "
          f"max {df.edge_buy.max():+.3f}")


if __name__ == "__main__":
    main()
