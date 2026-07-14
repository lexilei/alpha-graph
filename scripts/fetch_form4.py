"""Build data/cache/form4_trans.parquet: PIT insider-transaction base table.

Source: SEC Insider Transactions Data Sets (structured quarterly TSV bundles
extracted from Forms 3/4/5), https://www.sec.gov/data-research/sec-markets-data/
insider-transactions-data-sets . Quarterly ZIPs (2010q3 -> latest listed) are
downloaded once into data/raw/form4/ and kept; historical bundles are
immutable, so existing valid ZIPs are never re-downloaded.

Filters applied (this is the BASE TABLE for a future clustered-insider-purchase
factor, not the factor itself):
  - issuer CIK in our ~500-name corpus (authoritative cik/ticker map =
    distinct pairs in data/cache/filing_acceptance.parquet),
  - form 4 and 4/A only (amendments kept and flagged, NOT netted),
  - non-derivative table only (NONDERIV_TRANS.tsv),
  - transaction codes P and S only (open-market purchase/sale).

Join semantics: each NONDERIV_TRANS line is joined to SUBMISSION (issuer,
form, filing date) and to REPORTINGOWNER (owner CIK + role flags). Filings
with multiple reporting owners (joint Form 4s, a few percent) therefore
repeat each transaction line once per owner - the faithful representation
for counting distinct insiders. Exact duplicate rows are dropped; note this
also collapses same-filing lines identical across all kept columns (e.g.
two same-day lots at the same price whose only difference was a footnote).

Column-drift policy: headers are resolved against a per-table alias map; a
missing critical column raises loudly (listing the header found) instead of
guessing.

Output data/cache/form4_trans.parquet, one row per (transaction line,
reporting owner):
    cik                issuer CIK, 10-digit zero-padded (str)
    ticker             corpus ticker for that CIK
    accession          digits-only accession number (str)
    form               "4" or "4/A"
    is_amendment       form == "4/A"
    filing_date        official filing date (datetime)
    trans_date         transaction date (datetime, NaT if unparseable)
    trans_code         "P" or "S"
    shares             TRANS_SHARES (float)
    price              TRANS_PRICEPERSHARE (float, NaN if missing)
    dollar_value       shares * price (NaN if either missing)
    acquired_disposed  TRANS_ACQUIRED_DISP_CD ("A"/"D")
    direct_indirect    DIRECT_INDIRECT_OWNERSHIP ("D"/"I")
    owner_cik          reporting owner CIK, 10-digit zero-padded (str)
    is_officer         RPTOWNER_RELATIONSHIP contains "Officer"
    is_director        RPTOWNER_RELATIONSHIP contains "Director"
    is_tenpct          RPTOWNER_RELATIONSHIP contains "TenPercentOwner"

Usage:
    python scripts/fetch_form4.py             # skip if cache <7 days old
    python scripts/fetch_form4.py --refresh   # force rebuild (+ fetch any
                                              # newly listed quarters)
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from loguru import logger

from alpha_graph.config import CACHE_DIR, DATA_DIR

RAW_DIR = DATA_DIR / "raw" / "form4"
OUT_PATH = CACHE_DIR / "form4_trans.parquet"
CORPUS_PATH = CACHE_DIR / "filing_acceptance.parquet"
LANDING_URL = ("https://www.sec.gov/data-research/sec-markets-data/"
               "insider-transactions-data-sets")
USER_AGENT = "Lei Zhihan leizhihan0606@gmail.com"  # SEC fair-access identity
FIRST_QUARTER = "2010q3"
REQUEST_INTERVAL = 1.0  # polite: sequential, ~1 req/s for bulk files
MAX_AGE_DAYS = 7.0
MIN_FREE_GB = 20
DATE_FORMAT = "%d-%b-%Y"  # e.g. 31-AUG-2010 (both 2010q3 and 2026q2 vintages)

KEEP_FORMS = {"4", "4/A"}
KEEP_CODES = {"P", "S"}

# canonical name -> accepted header spellings (vintage drift). A critical
# column that resolves to none of its aliases raises instead of guessing.
TABLE_COLUMNS: dict[str, dict[str, tuple[str, ...]]] = {
    "SUBMISSION": {
        "ACCESSION_NUMBER": ("ACCESSION_NUMBER",),
        "FILING_DATE": ("FILING_DATE",),
        "DOCUMENT_TYPE": ("DOCUMENT_TYPE",),
        "ISSUERCIK": ("ISSUERCIK",),
    },
    "REPORTINGOWNER": {
        "ACCESSION_NUMBER": ("ACCESSION_NUMBER",),
        "RPTOWNERCIK": ("RPTOWNERCIK",),
        "RPTOWNER_RELATIONSHIP": ("RPTOWNER_RELATIONSHIP", "RPTOWNER_RLTNSHIP"),
    },
    "NONDERIV_TRANS": {
        "ACCESSION_NUMBER": ("ACCESSION_NUMBER",),
        "TRANS_DATE": ("TRANS_DATE",),
        "TRANS_CODE": ("TRANS_CODE",),
        "TRANS_SHARES": ("TRANS_SHARES",),
        "TRANS_PRICEPERSHARE": ("TRANS_PRICEPERSHARE",),
        "TRANS_ACQUIRED_DISP_CD": ("TRANS_ACQUIRED_DISP_CD",),
        "DIRECT_INDIRECT_OWNERSHIP": ("DIRECT_INDIRECT_OWNERSHIP",),
    },
}

_last_request_t = 0.0


# --------------------------------------------------------------------------- #
# HTTP (rate-limited, retry once on failure; optional 404 pass-through)
# --------------------------------------------------------------------------- #

def _get(session: requests.Session, url: str, ok404: bool = False) -> requests.Response | None:
    global _last_request_t
    for attempt in (0, 1):
        wait = REQUEST_INTERVAL - (time.monotonic() - _last_request_t)
        if wait > 0:
            time.sleep(wait)
        _last_request_t = time.monotonic()
        try:
            resp = session.get(url, timeout=120)
        except requests.RequestException as e:
            if attempt == 1:
                raise
            logger.warning(f"{url}: {e}; retrying in 5s")
            time.sleep(5)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 404 and ok404:
            return None
        if attempt == 1:
            resp.raise_for_status()
        backoff = float(resp.headers.get("Retry-After", 10))
        logger.warning(f"{url}: HTTP {resp.status_code}; retrying in {backoff:.0f}s")
        time.sleep(backoff)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Quarter discovery + download
# --------------------------------------------------------------------------- #

def check_disk_space() -> None:
    usage = shutil.disk_usage(DATA_DIR)
    free_gb = usage.free / 1e9
    logger.info(f"disk: {free_gb:.0f} GB free of {usage.total / 1e9:.0f} GB")
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"only {free_gb:.1f} GB free (<{MIN_FREE_GB} GB); "
                         "free up disk before downloading Form 4 bundles")


def list_quarters(session: requests.Session) -> dict[str, str]:
    """Scrape the landing page: quarter ('YYYYqN') -> absolute ZIP URL.

    The URL path prefix drifts across vintages (structureddata vs
    datastandardsinnovation), so we take the hrefs as published rather than
    hardcoding one pattern.
    """
    html = _get(session, LANDING_URL).text
    quarters: dict[str, str] = {}
    for m in re.finditer(r'href="([^"]*?(\d{4}q[1-4])_form345\.zip)"', html):
        quarters.setdefault(m.group(2), urljoin("https://www.sec.gov/", m.group(1)))
    if not quarters:
        raise RuntimeError("no *_form345.zip links found on the landing page; "
                           "layout changed?")
    wanted = {q: u for q, u in quarters.items() if q >= FIRST_QUARTER}
    logger.info(f"landing page lists {len(quarters)} quarters; "
                f"{len(wanted)} in range {FIRST_QUARTER}..{max(quarters)}")
    return dict(sorted(wanted.items()))


def _zip_ok(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            names = {Path(n).name.upper() for n in zf.namelist()}
        return {"SUBMISSION.TSV", "REPORTINGOWNER.TSV", "NONDERIV_TRANS.TSV"} <= names
    except (OSError, zipfile.BadZipFile):
        return False


def ensure_zips(session: requests.Session,
                quarters: dict[str, str]) -> tuple[list[Path], list[str]]:
    """Download any missing quarter ZIPs; return (paths, gap quarters).

    Gaps = quarters absent from the landing page within the wanted range,
    plus quarters whose download 404s (logged, not fatal).
    """
    expected = []
    y, q = int(FIRST_QUARTER[:4]), int(FIRST_QUARTER[-1])
    last = max(quarters)
    while f"{y}q{q}" <= last:
        expected.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    gaps = [e for e in expected if e not in quarters]
    if gaps:
        logger.warning(f"quarters missing from the landing page: {gaps}")

    paths: list[Path] = []
    for quarter, url in quarters.items():
        dest = RAW_DIR / f"{quarter}_form345.zip"
        if dest.exists() and _zip_ok(dest):
            paths.append(dest)
            continue
        if dest.exists():
            logger.warning(f"{dest.name}: existing file invalid; re-downloading")
        resp = _get(session, url, ok404=True)
        if resp is None:
            logger.error(f"{quarter}: HTTP 404 at {url}; skipping (gap)")
            gaps.append(quarter)
            continue
        tmp = dest.with_suffix(".zip.tmp")
        tmp.write_bytes(resp.content)
        if not _zip_ok(tmp):
            tmp.unlink(missing_ok=True)
            logger.error(f"{quarter}: downloaded file is not a valid form345 ZIP; "
                         "skipping (gap)")
            gaps.append(quarter)
            continue
        tmp.replace(dest)
        logger.info(f"{quarter}: downloaded {dest.stat().st_size / 1e6:.1f} MB")
        paths.append(dest)
    return paths, sorted(gaps)


# --------------------------------------------------------------------------- #
# TSV parsing (header-drift aware)
# --------------------------------------------------------------------------- #

def _read_table(zf: zipfile.ZipFile, table: str) -> pd.DataFrame:
    """Read one TSV from an open bundle with canonical column names (all str)."""
    members = {Path(n).name.upper(): n for n in zf.namelist()}
    member = members[f"{table}.TSV"]  # presence guaranteed by _zip_ok
    with zf.open(member) as fh:
        header = fh.readline().decode("utf-8", errors="replace")
    actual = [c.strip().lstrip("\ufeff") for c in header.rstrip("\r\n").split("\t")]

    resolved: dict[str, str] = {}  # actual header name -> canonical
    missing: list[str] = []
    for canonical, aliases in TABLE_COLUMNS[table].items():
        hit = next((a for a in aliases if a in actual), None)
        if hit is None:
            missing.append(f"{canonical} (accepted: {aliases})")
        else:
            resolved[hit] = canonical
    if missing:
        raise RuntimeError(
            f"{zf.filename}:{member}: critical column(s) not found: {missing}; "
            f"actual header = {actual}. Extend TABLE_COLUMNS aliases explicitly "
            "rather than guessing.")

    with zf.open(member) as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str, usecols=list(resolved),
                         quoting=csv.QUOTE_NONE, encoding="utf-8",
                         encoding_errors="replace", on_bad_lines="warn")
    return df.rename(columns=resolved)


def _norm_cik(s: pd.Series) -> pd.Series:
    return s.fillna("").str.strip().str.zfill(10)


def parse_quarter(zip_path: Path, cik_set: set[str]) -> pd.DataFrame:
    """One quarter -> joined P/S non-derivative rows for corpus issuers."""
    with zipfile.ZipFile(zip_path) as zf:
        sub = _read_table(zf, "SUBMISSION")
        sub = sub[sub["DOCUMENT_TYPE"].isin(KEEP_FORMS)].copy()
        sub["ISSUERCIK"] = _norm_cik(sub["ISSUERCIK"])
        sub = sub[sub["ISSUERCIK"].isin(cik_set)]
        sub = sub.drop_duplicates("ACCESSION_NUMBER", keep="first")
        if sub.empty:
            logger.warning(f"{zip_path.name}: 0 corpus form-4 filings")
            return pd.DataFrame()
        keep_acc = set(sub["ACCESSION_NUMBER"])

        trans = _read_table(zf, "NONDERIV_TRANS")
        trans = trans[trans["TRANS_CODE"].isin(KEEP_CODES)
                      & trans["ACCESSION_NUMBER"].isin(keep_acc)]

        owners = _read_table(zf, "REPORTINGOWNER")
        owners = owners[owners["ACCESSION_NUMBER"].isin(set(trans["ACCESSION_NUMBER"]))]

    df = (trans
          .merge(sub, on="ACCESSION_NUMBER", how="left")
          .merge(owners, on="ACCESSION_NUMBER", how="left"))
    logger.info(f"{zip_path.name}: {len(sub):,} corpus form-4 filings, "
                f"{len(trans):,} P/S trans lines -> {len(df):,} rows")
    return df


def build_table(zip_paths: list[Path], cik2ticker: dict[str, str]) -> pd.DataFrame:
    cik_set = set(cik2ticker)
    parts = [parse_quarter(p, cik_set) for p in zip_paths]
    raw = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    filing_date = pd.to_datetime(raw["FILING_DATE"], format=DATE_FORMAT, errors="coerce")
    if filing_date.isna().mean() > 0.001:
        raise RuntimeError(f"{filing_date.isna().sum():,} unparseable FILING_DATE "
                           f"values (format {DATE_FORMAT}); date format drifted?")
    trans_date = pd.to_datetime(raw["TRANS_DATE"], format=DATE_FORMAT, errors="coerce")
    if trans_date.isna().any():
        logger.warning(f"{trans_date.isna().sum():,} rows with missing/unparseable "
                       "TRANS_DATE kept as NaT")

    shares = pd.to_numeric(raw["TRANS_SHARES"], errors="coerce")
    price = pd.to_numeric(raw["TRANS_PRICEPERSHARE"], errors="coerce")
    rel = raw["RPTOWNER_RELATIONSHIP"].fillna("")
    out = pd.DataFrame({
        "cik": raw["ISSUERCIK"],
        "ticker": raw["ISSUERCIK"].map(cik2ticker),
        "accession": raw["ACCESSION_NUMBER"].str.replace(r"\D", "", regex=True),
        "form": raw["DOCUMENT_TYPE"],
        "is_amendment": raw["DOCUMENT_TYPE"].str.endswith("/A"),
        "filing_date": filing_date,
        "trans_date": trans_date,
        "trans_code": raw["TRANS_CODE"],
        "shares": shares,
        "price": price,
        "dollar_value": shares * price,
        "acquired_disposed": raw["TRANS_ACQUIRED_DISP_CD"],
        "direct_indirect": raw["DIRECT_INDIRECT_OWNERSHIP"],
        "owner_cik": _norm_cik(raw["RPTOWNERCIK"]),
        "is_officer": rel.str.contains("Officer", case=False),
        "is_director": rel.str.contains("Director", case=False),
        "is_tenpct": rel.str.contains("TenPercentOwner", case=False),
    })

    n0 = len(out)
    out = (out.drop_duplicates()
           .sort_values(["ticker", "trans_date", "accession", "owner_cik"])
           .reset_index(drop=True))
    logger.info(f"dropped {n0 - len(out):,} exact duplicate rows "
                f"({n0:,} -> {len(out):,})")
    return out


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #

def _clustering_preview(df: pd.DataFrame) -> Counter:
    """Raw event counts per year: (ticker, 21d window) with >=2 distinct
    officer/director owners making P buys. Onset-collapsed: a buy-day
    qualifies when the trailing 21 calendar days [t-20, t] contain >=2
    distinct officer/director buyers; a new event starts when a qualifying
    day is >20 days after the previous qualifying day for that ticker.
    Sizing preview only - NOT the factor.
    """
    buys = df[(df["trans_code"] == "P")
              & (df["is_officer"] | df["is_director"])
              & df["trans_date"].notna()]
    pairs = buys[["ticker", "owner_cik", "trans_date"]].drop_duplicates()
    events: Counter = Counter()
    for _, g in pairs.groupby("ticker"):
        dates = g["trans_date"].to_numpy()
        owners = g["owner_cik"].to_numpy()
        last_qual = None
        for t in np.sort(np.unique(dates)):
            mask = (dates > t - np.timedelta64(21, "D")) & (dates <= t)
            if len(set(owners[mask])) < 2:
                continue
            if last_qual is None or (t - last_qual) > np.timedelta64(20, "D"):
                events[pd.Timestamp(t).year] += 1
            last_qual = t
    return events


def validation_report(df: pd.DataFrame, gaps: list[str] | None) -> None:
    print("\n" + "=" * 64)
    print("  FORM 4 TRANSACTIONS - VALIDATION")
    print("=" * 64)
    print(f"\nrows: {len(df):,}  issuers: {df['cik'].nunique()}  "
          f"tickers: {df['ticker'].nunique()}  owners: {df['owner_cik'].nunique():,}")
    print("quarter gaps: " + ("not checked (cached run)" if gaps is None
                              else str(gaps) if gaps else "none"))
    multi = df.groupby("accession")["owner_cik"].transform("nunique") > 1
    print(f"rows from multi-owner filings (trans line repeated per owner): "
          f"{multi.sum():,} ({multi.mean() * 100:.2f}%)")
    print(f"amendments (4/A): {df['is_amendment'].sum():,} rows "
          f"({df['is_amendment'].mean() * 100:.2f}%)")

    print("\nrows per trans_date year:")
    per_year = df["trans_date"].dt.year.value_counts().sort_index()
    for yr, n in per_year.items():
        print(f"  {int(yr)}  {n:7,d}  {'#' * max(1, int(n / per_year.max() * 60))}")

    counts = df["trans_code"].value_counts()
    n_p, n_s = counts.get("P", 0), counts.get("S", 0)
    print(f"\ntrans_code: P {n_p:,}  S {n_s:,}  (S/P = {n_s / max(n_p, 1):.2f}x)")

    print("\nspot check - Dimon (owner CIK 0001195345) code-P buys of JPM")
    print("(a buy split across direct/trust lines counts at the filing level):")
    dimon = df[(df["ticker"] == "JPM") & (df["trans_code"] == "P")
               & (df["owner_cik"] == "0001195345")]
    for yr in (2012, 2016):
        d = dimon[dimon["trans_date"].dt.year == yr]
        by_acc = d.groupby("accession")["dollar_value"].sum()
        big_acc = by_acc[by_acc > 10e6]
        n_big_line = int((d["dollar_value"] > 10e6).sum())
        status = "PASS" if len(big_acc) else "FAIL"
        print(f"  {yr}: {len(d)} P lines; filings aggregating > $10M: {len(big_acc)} "
              f"(single lines > $10M: {n_big_line})  [{status}]")
        for acc, v in big_acc.items():
            sub = d[d["accession"] == acc]
            print(f"    {acc}: {sub['shares'].sum():,.0f} sh over {len(sub)} lines "
                  f"= ${v / 1e6:.1f}M ({sub['trans_date'].min().date()}"
                  f"..{sub['trans_date'].max().date()})")
    for t in ("NVDA", "AAPL"):
        s = df[(df["ticker"] == t) & (df["trans_code"] == "S")]
        yrs = s["trans_date"].dt.year
        print(f"  {t}: {len(s):,} S rows spanning {int(yrs.min())}-{int(yrs.max())} "
              f"({yrs.nunique()} distinct years)")

    p = df[df["trans_code"] == "P"]
    has_price = p["price"] > 0
    dv = p.loc[has_price, "dollar_value"]
    print(f"\nprice sanity (P rows): price>0 in {has_price.sum():,}/{len(p):,} "
          f"({has_price.mean() * 100:.1f}%)")
    print(f"  dollar_value p50 ${dv.quantile(0.5):,.0f}  p99 ${dv.quantile(0.99):,.0f}")

    print("\nclustering preview (NOT the factor): (ticker, 21d) events with >=2")
    print("distinct officer/director P-buyers, onset-collapsed, per year:")
    events = _clustering_preview(df)
    for yr in sorted(events):
        print(f"  {yr}  {events[yr]:4d}  {'#' * events[yr]}"[:80])
    print(f"  total: {sum(events.values())}")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch SEC insider-transactions data sets and build the "
                    "form-4 P/S base table for our corpus")
    p.add_argument("--refresh", action="store_true",
                   help=f"Force rebuild even if the parquet is <{MAX_AGE_DAYS:.0f} "
                        "days old (existing valid ZIPs are never re-downloaded; "
                        "delete files under data/raw/form4/ to re-fetch)")
    args = p.parse_args()

    age_days = ((time.time() - OUT_PATH.stat().st_mtime) / 86400
                if OUT_PATH.exists() else None)
    if not args.refresh and age_days is not None and age_days < MAX_AGE_DAYS:
        logger.info(f"{OUT_PATH.name} is {age_days:.1f} days old (<{MAX_AGE_DAYS:.0f}); "
                    "skipping fetch (--refresh to force)")
        validation_report(pd.read_parquet(OUT_PATH), gaps=None)
        return

    corpus = pd.read_parquet(CORPUS_PATH, columns=["cik", "ticker"]).drop_duplicates()
    cik2ticker = dict(zip(corpus["cik"], corpus["ticker"]))
    logger.info(f"corpus map: {len(cik2ticker)} CIKs, "
                f"{corpus['ticker'].nunique()} tickers ({CORPUS_PATH.name})")

    check_disk_space()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT,
                            "Accept-Encoding": "gzip, deflate"})
    quarters = list_quarters(session)
    zip_paths, gaps = ensure_zips(session, quarters)
    logger.info(f"{len(zip_paths)} quarter ZIPs on disk; gaps: {gaps if gaps else 'none'}")

    table = build_table(zip_paths, cik2ticker)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(OUT_PATH)
    logger.success(f"Saved {len(table):,} rows -> {OUT_PATH}")

    validation_report(table, gaps)


if __name__ == "__main__":
    main()
