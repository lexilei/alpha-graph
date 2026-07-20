#!/usr/bin/env python3
"""Fetch FINRA consolidated equity short interest (IDEAS.md I6; data
engineering, not a look).

Source: https://cdn.finra.org/equity/otcmarket/biweekly/shrt<YYYYMMDD>.csv
— pipe-delimited, one file per bi-monthly settlement cycle (mid-month and
month-end). CRITICAL coverage fact (finra.org files page, verified
2026-07-20): files BEFORE June 2021 contain OTC securities only;
exchange-listed names (our whole universe) appear from the 2021-06 cycle
onward, so the fetch starts there — the usable panel is ~5 years.

Settlement dates are exchange-adjusted, so for each cycle we try the
nominal date (15th / last calendar day) and walk back up to 4 calendar
days until a file exists.

PIT: FINRA disseminates roughly 7-9 business days after settlement; we
pin **avail_date = settlement + 10 US business days** — deliberately
conservative (errs stale, never early; the C17 availability philosophy).
Replace with the exact per-cycle dissemination calendar only as a
documented upgrade.

Output: data/cache/short_interest.parquet with columns
symbol, settlement_date, avail_date, short_qty, adv_qty, days_to_cover,
market_class. All rows kept (universe filtering happens at signal-build
time).

Usage: .venv/bin/python scripts/fetch_short_interest.py
Resumable: cycles already in the parquet are skipped.
"""
from __future__ import annotations

import io
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
OUT = CACHE / "short_interest.parquet"
URL = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{d}.csv"
START = date(2021, 6, 1)  # first consolidated (exchange-listed) cycle
AVAIL_LAG_BDAYS = 10

COLS = {
    "symbolCode": "symbol",
    "settlementDate": "settlement_date",
    "currentShortPositionQuantity": "short_qty",
    "averageDailyVolumeQuantity": "adv_qty",
    "daysToCoverQuantity": "days_to_cover",
    "marketClassCode": "market_class",
}


def nominal_cycle_dates(start: date, end: date) -> list[date]:
    """Mid-month (15th) and month-end nominal settlement dates."""
    out = []
    d = date(start.year, start.month, 1)
    while d <= end:
        mid = date(d.year, d.month, 15)
        nxt = (date(d.year + 1, 1, 1) if d.month == 12
               else date(d.year, d.month + 1, 1))
        eom = nxt - timedelta(days=1)
        out += [x for x in (mid, eom) if start <= x <= end]
        d = nxt
    return out


def fetch_cycle(nominal: date, session: requests.Session) -> pd.DataFrame | None:
    """Try nominal date then walk back up to 4 days; None if no file."""
    for back in range(5):
        d = nominal - timedelta(days=back)
        url = URL.format(d=d.strftime("%Y%m%d"))
        r = session.get(url, timeout=60)
        if r.status_code == 200 and r.text.startswith("accountingYearMonth"):
            df = pd.read_csv(io.StringIO(r.text), sep="|", dtype=str)
            df = df[list(COLS)].rename(columns=COLS)
            return df
    return None


def main() -> int:
    have: set[str] = set()
    frames = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        have = set(prev["settlement_date"].astype(str).str[:7].unique())
        frames.append(prev)
        print(f"resume: {len(have)} cycle-months already cached")

    session = requests.Session()
    end = date.today()
    new = 0
    for nominal in nominal_cycle_dates(START, end):
        # resume key: month + half (mid vs eom)
        got = fetch_cycle(nominal, session)
        if got is None:
            print(f"{nominal}: no file", file=sys.stderr)
            continue
        sd = got["settlement_date"].iloc[0]
        if frames and any(
            (f["settlement_date"] == sd).any() for f in frames
        ):
            continue
        got["avail_date"] = (
            pd.Timestamp(sd) + pd.tseries.offsets.BDay(AVAIL_LAG_BDAYS)
        ).date().isoformat()
        frames.append(got)
        new += 1
        print(f"{sd}: {len(got)} rows")
        time.sleep(0.5)

    if not frames:
        print("nothing fetched", file=sys.stderr)
        return 1
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["symbol", "settlement_date"], keep="last")
    for c in ("short_qty", "adv_qty", "days_to_cover"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} rows, "
          f"{out['settlement_date'].nunique()} cycles "
          f"({out['settlement_date'].min()} -> {out['settlement_date'].max()}), "
          f"{new} new")
    return 0


if __name__ == "__main__":
    sys.exit(main())
