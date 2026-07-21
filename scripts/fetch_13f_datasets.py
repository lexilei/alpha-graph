#!/usr/bin/env python3
"""Fetch SEC Form 13F structured data sets, filtered to the panel
(ownership-flow data engineering, not a look).

Source: https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets
— one zip per period (2013q2..2023q4 named `YYYYqN_form13f.zip`, 2024+
named `01mon2024-31mon2024_form13f.zip`), each containing SUBMISSION.tsv
(accession, CIK, filing date, period, type) and INFOTABLE.tsv (every
position of every filer: CUSIP, value, shares, put/call).

Per zip: stream-download → parse → keep positions whose CUSIP(8) maps to
a panel ticker (data/cache/cusip_ticker_map.parquet) → join filing
metadata → write data/cache/13f/<stem>.parquet → DELETE the zip. So the
~10GB of raw zips never accumulate. Resumable: existing parquets skip.

PIT: FILING_DATE is carried per row — availability discipline (first-
filed per (cik, period), amendment handling) is applied at SIGNAL build
time, not here; this store keeps 13F-HR and 13F-HR/A both, flagged.

Usage: .venv/bin/python scripts/fetch_13f_datasets.py
"""
from __future__ import annotations

import io
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
OUT_DIR = CACHE / "13f"
INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
UA = {"User-Agent": "Lei Zhihan leizhihan0606@gmail.com research"}
SUB_COLS = ["ACCESSION_NUMBER", "CIK", "FILING_DATE", "SUBMISSIONTYPE",
            "PERIODOFREPORT"]
INFO_COLS = ["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT",
             "SSHPRNAMTTYPE", "PUTCALL"]
CHUNK = 2_000_000


def list_zips(session: requests.Session) -> list[str]:
    html = session.get(INDEX, headers=UA, timeout=60).text
    hrefs = re.findall(r'href="([^"]*form13f\.zip)"', html)
    return ["https://www.sec.gov" + h if h.startswith("/") else h
            for h in dict.fromkeys(hrefs)]


def read_tsv(z: zipfile.ZipFile, name: str, usecols, chunks=False):
    member = next(n for n in z.namelist() if n.upper().endswith(name))
    kw = dict(sep="\t", dtype=str, encoding_errors="replace",
              on_bad_lines="skip",
              usecols=lambda c: c.upper() in usecols)
    if chunks:
        return pd.read_csv(z.open(member), chunksize=CHUNK, **kw)
    return pd.read_csv(z.open(member), **kw)


def process_zip(url: str, cusip_map: pd.DataFrame,
                session: requests.Session) -> None:
    stem = url.rsplit("/", 1)[1].replace(".zip", "")
    out = OUT_DIR / f"{stem}.parquet"
    if out.exists():
        return
    r = session.get(url, headers=UA, timeout=600)
    if r.status_code != 200 or r.content[:2] != b"PK":
        print(f"{stem}: download failed ({r.status_code})", file=sys.stderr)
        return
    keep8 = set(cusip_map["cusip8"])
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        sub = read_tsv(z, "SUBMISSION.TSV", set(SUB_COLS))
        sub.columns = [c.upper() for c in sub.columns]
        parts = []
        for chunk in read_tsv(z, "INFOTABLE.TSV", set(INFO_COLS), chunks=True):
            chunk.columns = [c.upper() for c in chunk.columns]
            chunk["cusip8"] = chunk["CUSIP"].str.strip().str[:8]
            parts.append(chunk[chunk["cusip8"].isin(keep8)])
    hits = pd.concat(parts, ignore_index=True)
    hits = hits.merge(sub, on="ACCESSION_NUMBER", how="left")
    hits = hits.merge(cusip_map[["cusip8", "ticker"]], on="cusip8", how="left")
    for c in ("VALUE", "SSHPRNAMT"):
        hits[c] = pd.to_numeric(hits[c], errors="coerce")
    OUT_DIR.mkdir(exist_ok=True)
    hits.to_parquet(out, index=False)
    print(f"{stem}: {len(hits)} panel positions / "
          f"{hits['ACCESSION_NUMBER'].nunique()} filings")


def main() -> int:
    cusip_map = pd.read_parquet(CACHE / "cusip_ticker_map.parquet")
    session = requests.Session()
    urls = list_zips(session)
    print(f"{len(urls)} dataset zips listed")
    for url in urls:
        try:
            process_zip(url, cusip_map, session)
        except Exception as e:  # noqa: BLE001 - log and continue, resumable
            print(f"{url}: ERROR {e}", file=sys.stderr)
        time.sleep(1.0)
    done = sorted(OUT_DIR.glob("*.parquet"))
    print(f"complete: {len(done)}/{len(urls)} period files in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
