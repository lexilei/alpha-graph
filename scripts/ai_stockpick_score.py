"""Score the AI stock-pick calibration experiment (see reports/ai_stockpick_prereg.md).

Reads every data/ai_stockpick/picks_*.json whose exit date has passed,
fetches adjusted closes (yfinance), and reports per round and cumulative:

  spread  = top5 equal-weight − bottom5 equal-weight over the holding week
  hit     = spread > 0
  IC      = Spearman(rank, realized 1w return)  [rank 1 = best expected]
  spy     = SPY same-window return (is the long side just beta?)

No judgment before 12 rounds, no conclusions before 24 (pinned in prereg).

Usage: .venv/bin/python scripts/ai_stockpick_score.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

PICKS = Path(__file__).resolve().parents[1] / "experiments" / "ai_stockpick"


def week_returns(tickers: list[str], entry: str, exit_: str) -> pd.Series:
    start = date.fromisoformat(entry)
    end = date.fromisoformat(exit_) + timedelta(days=4)  # holiday slack
    px = yf.download(tickers, start=str(start), end=str(end),
                     auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    # entry = first close on/after entry date, exit = first close on/after exit
    t0 = px.index[px.index >= pd.Timestamp(entry)][0]
    t1 = px.index[px.index >= pd.Timestamp(exit_)]
    if len(t1) == 0:
        raise SystemExit(f"exit date {exit_} not yet in price data")
    return px.loc[t1[0]] / px.loc[t0] - 1.0


def main() -> None:
    rows = []
    for f in sorted(PICKS.glob("picks_*.json")):
        p = json.loads(f.read_text())
        entry = p["pick_date"]
        exit_ = p["exit"].split()[0]
        if date.fromisoformat(exit_) > date.today():
            print(f"{f.name}: exit {exit_} in the future, skipped")
            continue
        tickers = [r["ticker"] for r in p["ranking"]]
        ret = week_returns(tickers + ["SPY"], entry, exit_)
        long5 = ret[p["long5"]].mean()
        short5 = ret[p["short5"]].mean()
        ranks = {r["ticker"]: r["rank"] for r in p["ranking"]}
        ic, _ = spearmanr([-ranks[t] for t in tickers],
                          [ret[t] for t in tickers])
        rows.append({"round": p["round"], "date": entry,
                     "long5": long5, "short5": short5,
                     "spread": long5 - short5, "hit": long5 - short5 > 0,
                     "ic": ic, "spy": ret["SPY"]})
    if not rows:
        print("no scorable rounds yet")
        return
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    n = len(df)
    print(f"\nrounds {n} | hit {df.hit.mean():.0%} | "
          f"mean spread {df.spread.mean():+.4f} | mean IC {df.ic.mean():+.3f}")
    if n < 12:
        print("n < 12: below pre-registered look threshold — direction only")
    elif n < 24:
        print("12 <= n < 24: direction readable, no conclusions (prereg)")


if __name__ == "__main__":
    main()
