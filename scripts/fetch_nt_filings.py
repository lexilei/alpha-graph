"""Fetch NT 10-K / NT 10-Q rows (Form 12b-25 late-filing notifications) for
the C18 `nt_filing_veto` candidate -> data/cache/nt_filings.parquet.

NT forms are absent from the filing corpus (never downloaded), so this walks
the same SEC submissions API as fetch_acceptance_datetimes.py — filings.recent
plus archived pages — for every CIK in data/cache/filing_acceptance.parquet's
(cik, ticker) map, keeping rows whose normalized form (strip + upper) starts
with "NT 10-K" or "NT 10-Q" (covers /A amendments and the NT 10-KT/QT
transition-period variants).

MIN_DATE = 2008-01-01: the earliest NT that can open an in-sample veto window
is filed 2011-04-05 (the market_data calendar starts 2011-04-06 and the C18
window starts at filing_date + 1 calendar day), and the pinned first-or-rare
rule looks back 3 years from a trigger -> 2008-04-05; 2008-01-01 adds margin.
Archived pages entirely before MIN_DATE are skipped and parsed rows older
than it are dropped, so the cache has a clean documented boundary.

Universe caveat (inherited from the corpus): only the CIKs backing the
current panel tickers are queried. Index members that departed before the
panel was built — historically the likelier late filers — are not fetchable
here; the C18 evaluation must state this.

Shared plumbing is imported from scripts/fetch_acceptance_datetimes.py:
_get_json (rate-limited <=8 req/s, retry-once) and _to_eastern (the API's
UTC -> tz-naive ET wall-time conversion including the mislabeled-ET quirk
rules A/B/C; with the few NT rows per CIK the filer-level rule C cannot
fire, and acceptance_ts is informational for C18 anyway — the veto window is
keyed to filing_date + 1 calendar day, not to acceptance time).

Output columns (filing_acceptance-compatible):
    accession, cik, ticker, form, filing_date, acceptance_ts,
    tz_reinterpreted, source

Usage:
    python scripts/fetch_nt_filings.py             # skip if cache <7 days old
    python scripts/fetch_nt_filings.py --refresh   # force re-fetch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_acceptance_datetimes import (  # noqa: E402
    SUBMISSIONS_BASE,
    USER_AGENT,
    _get_json,
    _to_eastern,
)

from alpha_graph.config import CACHE_DIR  # noqa: E402

ACCEPTANCE_PATH = CACHE_DIR / "filing_acceptance.parquet"
OUT_PATH = CACHE_DIR / "nt_filings.parquet"
NT_PREFIXES = ("NT 10-K", "NT 10-Q")
MIN_DATE = "2008-01-01"
MAX_AGE_DAYS = 7.0

COLUMNS = ["accession", "cik", "ticker", "form", "filing_date",
           "acceptance_ts", "tz_reinterpreted", "source"]


def _is_nt(form: object) -> bool:
    return str(form).strip().upper().startswith(NT_PREFIXES)


def _parse_block_nt(block: dict, cik: str, ticker: str) -> list[dict]:
    """Parallel-array walk (fetch_acceptance_datetimes._parse_block) keeping
    NT rows instead of KEEP_FORMS rows; the form is stored normalized."""
    accs = block.get("accessionNumber", [])
    forms = block.get("form", [])
    fdates = block.get("filingDate", [])
    adts = block.get("acceptanceDateTime", [])
    return [
        {
            "accession": accs[i].replace("-", ""),
            "cik": cik,
            "ticker": ticker,
            "form": str(forms[i]).strip().upper(),
            "filing_date": fdates[i],
            "acceptance_raw": adts[i] if i < len(adts) else None,
        }
        for i in range(len(accs))
        if _is_nt(forms[i])
    ]


def fetch_cik_nt(session: requests.Session, cik: str, ticker: str) -> list[dict]:
    """filings.recent + every archived page not entirely before MIN_DATE.

    Unlike fetch_acceptance_datetimes.fetch_cik there is no early stop: we do
    not know which accessions exist — NT rows are whatever turns up."""
    j = _get_json(session, f"{SUBMISSIONS_BASE}CIK{cik}.json")
    rows = _parse_block_nt(j.get("filings", {}).get("recent", {}), cik, ticker)
    for page in j.get("filings", {}).get("files", []):
        if page.get("filingTo", "9999-99-99") < MIN_DATE:
            continue
        pj = _get_json(session, SUBMISSIONS_BASE + page["name"])
        block = pj.get("filings", {}).get("recent", pj)  # archive pages are bare blocks
        rows.extend(_parse_block_nt(block, cik, ticker))
    return rows


def build_table() -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})

    pairs = (pd.read_parquet(ACCEPTANCE_PATH, columns=["cik", "ticker"])
             .drop_duplicates().sort_values(["cik", "ticker"]))
    logger.info(f"CIK universe: {len(pairs)} (cik, ticker) pairs from "
                f"{ACCEPTANCE_PATH.name}")

    all_rows: list[dict] = []
    for i, r in enumerate(pairs.itertuples(), 1):
        try:
            all_rows.extend(fetch_cik_nt(session, r.cik, r.ticker))
        except Exception as e:  # noqa: BLE001
            logger.error(f"CIK {r.cik} ({r.ticker}): fetch failed: {e}")
        if i % 50 == 0:
            logger.info(f"{i}/{len(pairs)} CIKs fetched ({len(all_rows)} NT rows)")

    if not all_rows:
        logger.warning("no NT rows found in the entire universe")
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(all_rows).drop_duplicates("accession", keep="first")
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df[df["filing_date"] >= pd.Timestamp(MIN_DATE)].reset_index(drop=True)
    df["acceptance_ts"], df["tz_reinterpreted"] = _to_eastern(
        df["acceptance_raw"], df["filing_date"], df["cik"])
    df["source"] = "sec_submissions_api"
    return df[COLUMNS].sort_values(["ticker", "filing_date"]).reset_index(drop=True)


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("  NT FILINGS (Form 12b-25) — FETCH REPORT")
    print("=" * 64)
    print(f"\nrows: {len(df)}  tickers: {df['ticker'].nunique() if len(df) else 0}  "
          f"window: {MIN_DATE} -> today")
    if not len(df):
        print("=" * 64)
        return
    print("\nby form:")
    for form, n in df["form"].value_counts().items():
        print(f"  {form:12s} {n:4d}")
    print("\nby year:")
    for year, n in df["filing_date"].dt.year.value_counts().sort_index().items():
        print(f"  {year}  {n:4d}")
    print("\nby ticker (with dates):")
    for t, g in df.groupby("ticker"):
        dates = ", ".join(f"{d.date()} {f}" for d, f in zip(g["filing_date"], g["form"]))
        print(f"  {t:6s} {dates}")
    print("=" * 64)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch NT 10-K/10-Q rows for the corpus CIKs")
    p.add_argument("--refresh", action="store_true",
                   help=f"Force re-fetch even if the parquet is <{MAX_AGE_DAYS:.0f} days old")
    args = p.parse_args()

    age_days = ((time.time() - OUT_PATH.stat().st_mtime) / 86400 if OUT_PATH.exists() else None)
    if not args.refresh and age_days is not None and age_days < MAX_AGE_DAYS:
        logger.info(f"{OUT_PATH.name} is {age_days:.1f} days old (<{MAX_AGE_DAYS:.0f}); "
                    "skipping fetch (--refresh to force)")
        df = pd.read_parquet(OUT_PATH)
    else:
        df = build_table()
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_PATH.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(OUT_PATH)
        logger.success(f"Saved {len(df)} rows -> {OUT_PATH}")

    report(df)


if __name__ == "__main__":
    main()
