#!/usr/bin/env python3
"""Fetch dividend histories (ex-date + amount) for the panel universe
(IDEAS.md I22 prerequisite; data engineering, not a look).

Source: yfinance actions (same provider as market.py's prices — shared
caveat: ex-dates and amounts are as-currently-known, not point-in-time
announcements. For I22 this is acceptable at REGISTRATION time only for
the ex-date CALENDAR (ex-dates are scheduled and stable); any
announcement-timing construction would need a PIT source and its own
pin. The amount series is back-adjusted by yfinance splits.)

Output: data/cache/dividends.parquet (ticker, ex_date, amount).

Usage: .venv/bin/python scripts/fetch_dividends.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
OUT = CACHE / "dividends.parquet"


def main() -> int:
    market = pd.read_parquet(CACHE / "market_data.parquet")
    tickers = sorted(market["ticker"].unique())
    print(f"{len(tickers)} tickers")
    rows = []
    for i, t in enumerate(tickers):
        try:
            div = yf.Ticker(t).dividends
        except Exception as e:  # noqa: BLE001 - provider errors are data
            print(f"{t}: ERROR {e}", file=sys.stderr)
            continue
        if div is None or div.empty:
            continue
        d = div.reset_index()
        d.columns = ["ex_date", "amount"]
        d["ticker"] = t
        rows.append(d)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(tickers)}")
    out = pd.concat(rows, ignore_index=True)
    out["ex_date"] = pd.to_datetime(out["ex_date"]).dt.tz_localize(None)
    out = out[["ticker", "ex_date", "amount"]].sort_values(["ticker", "ex_date"])
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} rows / {out['ticker'].nunique()} tickers, "
          f"{out['ex_date'].min().date()} -> {out['ex_date'].max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
