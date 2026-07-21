#!/usr/bin/env python3
"""Build a CUSIP(8) → panel-ticker map from SEC fails-to-deliver files
(13F pipeline prerequisite; data engineering, not a look).

FTD files (pipe-delimited: SETTLEMENT DATE|CUSIP|SYMBOL|...) carry the
CUSIP↔symbol pairing for every symbol with at least one fail in the
half-month — over many vintages virtually every listed name appears.
Multiple vintages 2014→now also catch CUSIP changes (corporate actions).

Matching: first 8 CUSIP chars (issue level, drops the check digit).
Symbol normalization: spaces/dots → '-' (FTD prints class shares as
'BRK B'/'BRK.B'; the panel uses 'BRK-B').

Output: data/cache/cusip_ticker_map.parquet (cusip8, ticker, first_seen,
last_seen) + a coverage report vs the panel.

Usage: .venv/bin/python scripts/build_cusip_map.py
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
OUT = CACHE / "cusip_ticker_map.parquet"
UA = {"User-Agent": "Lei Zhihan leizhihan0606@gmail.com research"}
PATHS = ("files/data/fails-deliver-data", "files/data/other/fails-deliver-data")
# two vintages per year, first half-month files
YEARS = range(2014, 2027)
MONTHS = ("03", "09")


COMPRESSED_CLASS = {"BRKB": "BRK-B", "BFB": "BF-B"}  # FINRA/FTD compressed forms


def norm_symbol(s: str) -> str:
    t = s.strip().replace(" ", "-").replace(".", "-").replace("/", "-")
    return COMPRESSED_CLASS.get(t, t)


def fetch_vintage(yyyymm: str, session: requests.Session) -> pd.DataFrame | None:
    for prefix in PATHS:
        url = f"https://www.sec.gov/{prefix}/cnsfails{yyyymm}a.zip"
        r = session.get(url, headers=UA, timeout=120)
        if r.status_code != 200 or not r.content[:2] == b"PK":
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = z.namelist()[0]
            df = pd.read_csv(
                z.open(name), sep="|", dtype=str,
                usecols=lambda c: c.upper() in ("CUSIP", "SYMBOL"),
                on_bad_lines="skip", encoding_errors="replace",
            )
        df.columns = [c.upper() for c in df.columns]
        df = df.dropna().drop_duplicates()
        df["vintage"] = yyyymm
        return df
    return None


def main() -> int:
    market = pd.read_parquet(CACHE / "market_data.parquet")
    panel = set(market["ticker"].unique())
    session = requests.Session()
    frames = []
    for y in YEARS:
        for m in MONTHS:
            got = fetch_vintage(f"{y}{m}", session)
            print(f"{y}{m}: {'ok ' + str(len(got)) if got is not None else 'MISS'}",
                  file=sys.stderr)
            if got is not None:
                frames.append(got)
            time.sleep(0.5)
    allp = pd.concat(frames, ignore_index=True)
    allp["ticker"] = allp["SYMBOL"].map(norm_symbol)
    allp = allp[allp["ticker"].isin(panel)]
    allp["cusip8"] = allp["CUSIP"].str.strip().str[:8]
    g = allp.groupby(["cusip8", "ticker"])["vintage"].agg(["min", "max"]).reset_index()
    g.columns = ["cusip8", "ticker", "first_seen", "last_seen"]
    # a cusip8 must map to exactly one panel ticker; drop ambiguous
    dup = g["cusip8"].duplicated(keep=False)
    if dup.any():
        print(f"ambiguous cusip8 dropped: {g[dup].to_dict('records')}",
              file=sys.stderr)
        g = g[~dup]
    g.to_parquet(OUT, index=False)
    covered = set(g["ticker"])
    missing = sorted(panel - covered)
    print(f"wrote {OUT}: {len(g)} cusip8 rows, {len(covered)}/{len(panel)} "
          f"panel tickers covered; missing: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
