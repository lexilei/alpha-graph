"""Build data/cache/filing_acceptance.parquet: accession -> true ACCEPTANCE-DATETIME.

Our filing JSONs (data/filings/<TICKER>/8-K_YYYY-MM-DD_ACCESSION.json) only carry
the official filing_date from the filename. What determines whether a signal was
actually tradeable is the EDGAR acceptance datetime: filings accepted after
~17:30 ET are disseminated the next business day (and get the next day's filing
date), and anything accepted after 16:00 ET cannot be traded at that day's close.
This script fetches acceptance datetimes for every CIK in our corpus from the
SEC submissions API and saves a lookup table for the live-availability data gate.

TIMEZONE NOTE (verified against raw EDGAR headers): the submissions API serves
`acceptanceDateTime` in genuine UTC (e.g. AAPL 10-K 0000320193-19-000119 is
"2019-10-30T22:12:36.000Z" in the API vs "2019-10-30 18:12:36" ET on the EDGAR
index page — EDT is UTC-4). We convert UTC -> America/New_York (DST-aware) and
store `acceptance_ts` as a tz-naive US/Eastern wall-time datetime, because the
16:00 close / 17:30 dissemination cutoffs are ET wall-clock rules.

API QUIRK: some rows carry the ET wall time with a spurious "Z" (23 spot-checks
against EDGAR index pages confirm raw == API value on such rows, e.g. DAL
0000027904-26-000029: API "06:30:10Z", EDGAR "06:30:10" ET). In our universe
this is concentrated in a handful of filers (INTC, MSCI, GEN, ISRG, SW) whose
ENTIRE submissions history is mislabeled — 11/11 sampled rows shifted exactly
4/5h, corrections span every year they filed, and none of them shows the
14:00-17:00 ET mass every real filer has. Three corrections reinterpret the raw
value as ET, marked tz_reinterpreted=True:
  (A) window: UTC-parsed ET time outside EDGAR's 06:00-22:05 acceptance window
      while the raw wall time is inside it;
  (B) dissemination consistency: UTC-parsed time before 17:30 but filing_date
      1-4 days AFTER the acceptance date (impossible for a pre-cutoff
      acceptance) while the raw wall time is >= 17:20 (consistent with the
      next-business-day rule);
  (C) filer-level: CIKs with >=5 A/B detections, >=5% of their rows detected,
      and (the signature) near-zero undetected 14:00-16:59 mass get ALL rows
      reinterpreted (in-window raw values only).
Mislabeled rows always shift times EARLIER, so any residual undetected cases
(isolated true 10:00-17:29 acceptances stored 4-5h early at a non-flagged
filer; audit found 1 in 39 settled-history samples, a filing <7 days old)
bias the after-16:00-ET fraction DOWN, by well under 0.5pp.

Output columns:
    accession     digits-only accession number (str, joins to our filenames)
    cik           10-digit zero-padded CIK (str)
    ticker        corpus ticker the CIK resolved from (first alphabetically)
    form          8-K / 10-K / 10-Q and their /A amendments
    filing_date   official filing date (datetime, midnight)
    acceptance_ts EDGAR acceptance datetime, tz-naive US/Eastern wall time
    tz_reinterpreted  True where the API's mislabeled-ET quirk was corrected
    source        "sec_submissions_api"

Usage:
    python scripts/fetch_acceptance_datetimes.py             # skip if cache <7 days old
    python scripts/fetch_acceptance_datetimes.py --refresh   # force re-fetch
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import pandas as pd
import requests
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR

OUT_PATH = CACHE_DIR / "filing_acceptance.parquet"
SUBMISSIONS_BASE = "https://data.sec.gov/submissions/"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "Lei Zhihan leizhihan0606@gmail.com"  # SEC fair-access identity
KEEP_FORMS = {"8-K", "10-K", "10-Q", "8-K/A", "10-K/A", "10-Q/A"}
REQUEST_INTERVAL = 0.13  # ~8 req/s, SEC's stated fair-access ceiling is 10
MAX_AGE_DAYS = 7.0

_last_request_t = 0.0


# --------------------------------------------------------------------------- #
# HTTP (rate-limited, retry once on 429/5xx)
# --------------------------------------------------------------------------- #

def _get_json(session: requests.Session, url: str) -> dict:
    global _last_request_t
    for attempt in (0, 1):
        wait = REQUEST_INTERVAL - (time.monotonic() - _last_request_t)
        if wait > 0:
            time.sleep(wait)
        _last_request_t = time.monotonic()
        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException as e:
            if attempt == 1:
                raise
            logger.warning(f"{url}: {e}; retrying in 2s")
            time.sleep(2)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == 1:
                resp.raise_for_status()
            backoff = float(resp.headers.get("Retry-After", 5))
            logger.warning(f"{url}: HTTP {resp.status_code}; retrying in {backoff:.0f}s")
            time.sleep(backoff)
            continue
        resp.raise_for_status()
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Corpus enumeration
# --------------------------------------------------------------------------- #

def enumerate_corpus() -> pd.DataFrame:
    """One row per filing JSON on disk: ticker, form, filing_date, accession."""
    rows = []
    for tdir in sorted(FILINGS_DIR.iterdir()):
        if not tdir.is_dir() or tdir.name.startswith((".", "_")):
            continue
        for f in tdir.glob("*_*_*.json"):
            # filename: {form}_{YYYY-MM-DD}_{accession}.json (form may contain '-')
            parts = f.stem.split("_")
            acc = "".join(ch for ch in parts[-1] if ch.isdigit())
            rows.append((tdir.name, parts[0], parts[1], acc))
    df = pd.DataFrame(rows, columns=["ticker", "form", "filing_date", "accession"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


# --------------------------------------------------------------------------- #
# Ticker -> CIK
# --------------------------------------------------------------------------- #

def resolve_ciks(session: requests.Session, tickers: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map ticker -> 10-digit CIK via company_tickers.json; edgartools fallback."""
    data = _get_json(session, COMPANY_TICKERS_URL)
    sec_map = {row["ticker"].upper(): f"{int(row['cik_str']):010d}" for row in data.values()}

    resolved: dict[str, str] = {}
    misses: list[str] = []
    for t in tickers:
        for cand in (t, t.replace(".", "-"), t.replace("-", ".")):
            if cand.upper() in sec_map:
                resolved[t] = sec_map[cand.upper()]
                break
        else:
            misses.append(t)

    if misses:  # renames/delistings: try edgartools' reference data as a fallback
        try:
            from alpha_graph.data.filings import _init_edgar
            _init_edgar()
            from edgar import Company
            for t in list(misses):
                try:
                    resolved[t] = f"{int(Company(t).cik):010d}"
                    misses.remove(t)
                    logger.info(f"{t}: resolved via edgartools fallback")
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            logger.warning(f"edgartools fallback unavailable: {e}")
    return resolved, misses


# --------------------------------------------------------------------------- #
# Submissions API
# --------------------------------------------------------------------------- #

def _parse_block(block: dict, cik: str, ticker: str) -> list[dict]:
    """Parse one parallel-array filings block, keeping only KEEP_FORMS rows."""
    accs = block.get("accessionNumber", [])
    forms = block.get("form", [])
    fdates = block.get("filingDate", [])
    adts = block.get("acceptanceDateTime", [])
    return [
        {
            "accession": accs[i].replace("-", ""),
            "cik": cik,
            "ticker": ticker,
            "form": forms[i],
            "filing_date": fdates[i],
            "acceptance_raw": adts[i] if i < len(adts) else None,
        }
        for i in range(len(accs))
        if forms[i] in KEEP_FORMS
    ]


def fetch_cik(session: requests.Session, cik: str, ticker: str,
              want: set[str], min_date: str) -> list[dict]:
    """Fetch filings.recent plus whichever archive pages are needed to cover
    `want` (our corpus accessions for this CIK); skip pages entirely before
    `min_date`, and stop early once every wanted accession has been seen."""
    j = _get_json(session, f"{SUBMISSIONS_BASE}CIK{cik}.json")
    rows = _parse_block(j.get("filings", {}).get("recent", {}), cik, ticker)
    have = {r["accession"] for r in rows}

    pages = sorted(j.get("filings", {}).get("files", []),
                   key=lambda p: p.get("filingTo", ""), reverse=True)
    for page in pages:
        if want <= have:
            break
        if page.get("filingTo", "9999-99-99") < min_date:
            continue
        pj = _get_json(session, SUBMISSIONS_BASE + page["name"])
        block = pj.get("filings", {}).get("recent", pj)  # archive pages are bare blocks
        page_rows = _parse_block(block, cik, ticker)
        rows.extend(page_rows)
        have.update(r["accession"] for r in page_rows)
    return rows


def prefix_cik_pass(session: requests.Session, corpus: pd.DataFrame,
                    fetched: set[str]) -> list[dict]:
    """Recover tickers whose current-ticker CIK misses the corpus filings (CIK
    migrations, e.g. XOM's 2026 holdco reorg re-pointed the ticker to a new
    CIK). For each ticker with >=3 unmatched accessions sharing a majority
    accession-number prefix (first 10 digits = transmitting filer's CIK,
    which is the company itself for self-filers), fetch that CIK too. Safe:
    rows only ever match by exact accession."""
    rows: list[dict] = []
    unmatched = corpus[~corpus["accession"].isin(fetched)]
    for ticker, grp in unmatched.groupby("ticker"):
        prefixes = grp["accession"].str[:10].value_counts()
        cik, n = prefixes.index[0], int(prefixes.iloc[0])
        if n < 3:
            continue
        logger.info(f"{ticker}: {len(grp)} unmatched; retrying via accession-prefix CIK {cik}")
        try:
            rows.extend(fetch_cik(session, cik, ticker, set(grp["accession"]),
                                  str(grp["filing_date"].min().date())))
        except Exception as e:  # noqa: BLE001
            logger.error(f"prefix CIK {cik} ({ticker}): fetch failed: {e}")
    return rows


def _to_eastern(raw: pd.Series, filing_date: pd.Series,
                cik: pd.Series) -> tuple[pd.Series, pd.Series]:
    """API UTC -> tz-naive ET wall time, correcting mislabeled-ET rows (see
    module docstring, rules A/B/C). Returns (acceptance_ts, tz_reinterpreted)."""
    utc_et = (pd.to_datetime(raw, utc=True, errors="coerce", format="mixed")
              .dt.tz_convert("America/New_York").dt.tz_localize(None))
    wall = pd.to_datetime(raw.str.replace(r"[Zz]|\.\d+", "", regex=True), errors="coerce")

    day_min = utc_et.dt.hour * 60 + utc_et.dt.minute
    wall_min = wall.dt.hour * 60 + wall.dt.minute
    wall_in_window = wall_min.between(6 * 60, 22 * 60 + 5)
    rule_a = ~day_min.between(6 * 60, 22 * 60 + 5) & wall_in_window
    date_gap = (filing_date - utc_et.dt.normalize()).dt.days
    rule_b = ((day_min < 17 * 60 + 30) & date_gap.between(1, 4)
              & (wall_min >= 17 * 60 + 20) & wall_in_window)
    fix = (rule_a | rule_b) & utc_et.notna()

    # Rule C: filers whose whole history is mislabeled (see docstring).
    per = pd.DataFrame({
        "det": fix,
        "evening": ~fix & day_min.between(14 * 60, 16 * 60 + 59),
    }).groupby(cik).agg(n=("det", "size"), d=("det", "sum"), ev=("evening", "sum"))
    bad = per.index[(per.d >= 5) & (per.d / per.n >= 0.05) & (per.ev <= 0.05 * per.d)]
    rule_c = cik.isin(bad) & ~fix & wall_in_window & utc_et.notna()
    fix = fix | rule_c
    logger.info(f"tz quirk corrections: {rule_a.sum()} out-of-window, "
                f"{(rule_b & ~rule_a).sum()} dissemination-inconsistent, "
                f"{rule_c.sum()} filer-level (CIKs: {sorted(bad)})")
    return utc_et.where(~fix, wall), fix


def build_table(corpus: pd.DataFrame) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})

    tickers = sorted(corpus["ticker"].unique())
    resolved, misses = resolve_ciks(session, tickers)
    if misses:
        logger.warning(f"{len(misses)} tickers failed CIK resolution: {misses}")
    logger.info(f"Resolved {len(resolved)}/{len(tickers)} tickers "
                f"to {len(set(resolved.values()))} unique CIKs")

    # A CIK can back several tickers (share classes); fetch each CIK once.
    cik_tickers: dict[str, list[str]] = defaultdict(list)
    for t in sorted(resolved):
        cik_tickers[resolved[t]].append(t)
    corpus_by_ticker = dict(tuple(corpus.groupby("ticker")))

    all_rows: list[dict] = []
    for i, (cik, ts) in enumerate(sorted(cik_tickers.items()), 1):
        sub = pd.concat([corpus_by_ticker[t] for t in ts])
        want = set(sub["accession"])
        min_date = str(sub["filing_date"].min().date())
        try:
            all_rows.extend(fetch_cik(session, cik, ts[0], want, min_date))
        except Exception as e:  # noqa: BLE001
            logger.error(f"CIK {cik} ({ts[0]}): fetch failed: {e}")
        if i % 50 == 0:
            logger.info(f"{i}/{len(cik_tickers)} CIKs fetched ({len(all_rows):,} rows)")

    all_rows.extend(prefix_cik_pass(session, corpus, {r["accession"] for r in all_rows}))

    df = pd.DataFrame(all_rows).drop_duplicates("accession", keep="first")
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["acceptance_ts"], df["tz_reinterpreted"] = _to_eastern(
        df["acceptance_raw"], df["filing_date"], df["cik"])
    df["source"] = "sec_submissions_api"
    return df[["accession", "cik", "ticker", "form", "filing_date",
               "acceptance_ts", "tz_reinterpreted", "source"]]


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #

def validation_report(corpus: pd.DataFrame, table: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("  FILING ACCEPTANCE — VALIDATION")
    print("=" * 64)

    merged = corpus.merge(
        table[["accession", "acceptance_ts", "tz_reinterpreted"]], on="accession", how="left"
    )
    matched = merged["acceptance_ts"].notna()
    print(f"\ncoverage: {matched.sum():,}/{len(merged):,} corpus filings matched "
          f"({matched.mean() * 100:.2f}%)")
    if (~matched).any():
        print("  sample misses (up to 20):")
        for _, r in merged[~matched].head(20).iterrows():
            print(f"    {r.ticker:6s} {r.form:5s} {r.filing_date.date()} {r.accession}")

    m = merged[matched].copy()
    print(f"tz-quirk corrected rows among matches: {int(m['tz_reinterpreted'].sum()):,} "
          f"({m['tz_reinterpreted'].mean() * 100:.2f}%)")
    delta = (m["acceptance_ts"].dt.normalize() - m["filing_date"]).dt.days
    ok = delta.between(-1, 1)
    print(f"\nsanity: acceptance date within filing_date +/- 1 day: "
          f"{ok.mean() * 100:.3f}% ({(~ok).sum()} outside)")
    prev_evening = (delta == -1).sum()
    print(f"  accepted the evening BEFORE the filing date (post-17:30 spillover): "
          f"{prev_evening:,} ({prev_evening / len(m) * 100:.2f}%)")
    weekend = (delta.between(-4, -2) & (m["acceptance_ts"].dt.weekday == 4)).sum()
    print(f"  accepted Friday evening -> Monday filing date (weekend spillover, "
          f"expected): {weekend:,}")
    outliers = m[~ok & ~(delta.between(-4, -2) & (m["acceptance_ts"].dt.weekday == 4))]
    if len(outliers):
        print(f"  unexplained outliers: {len(outliers)} (up to 5):")
        for _, r in outliers.head(5).iterrows():
            print(f"    {r.ticker:6s} {r.form:5s} filed {r.filing_date.date()} "
                  f"accepted {r.acceptance_ts}")

    k8 = m[m["form"].str.startswith("8-K")]
    print(f"\n8-K acceptance hour-of-day (ET, n={len(k8):,}):")
    hours = k8["acceptance_ts"].dt.hour.value_counts().sort_index()
    for h, n in hours.items():
        print(f"  {h:02d}:00  {n:6,d}  {'#' * max(1, int(n / len(k8) * 120))}")

    mins = k8["acceptance_ts"].dt.hour * 60 + k8["acceptance_ts"].dt.minute
    after_close = (mins >= 16 * 60).mean()
    after_cutoff = (mins >= 17 * 60 + 30).mean()
    print(f"\n8-Ks accepted at/after 16:00 ET (NOT tradeable at same close): "
          f"{after_close * 100:.2f}%")
    print(f"8-Ks accepted at/after 17:30 ET (disseminated next business day): "
          f"{after_cutoff * 100:.2f}%")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Fetch SEC acceptance datetimes for our filing corpus")
    p.add_argument("--refresh", action="store_true",
                   help=f"Force re-fetch even if the parquet is <{MAX_AGE_DAYS:.0f} days old")
    args = p.parse_args()

    corpus = enumerate_corpus()
    logger.info(f"Corpus: {len(corpus):,} filings, {corpus['ticker'].nunique()} tickers, "
                f"{corpus['filing_date'].min().date()} -> {corpus['filing_date'].max().date()}")

    age_days = ((time.time() - OUT_PATH.stat().st_mtime) / 86400 if OUT_PATH.exists() else None)
    if not args.refresh and age_days is not None and age_days < MAX_AGE_DAYS:
        logger.info(f"{OUT_PATH.name} is {age_days:.1f} days old (<{MAX_AGE_DAYS:.0f}); "
                    "skipping fetch (--refresh to force)")
        table = pd.read_parquet(OUT_PATH)
    else:
        table = build_table(corpus)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_PATH.with_suffix(".parquet.tmp")
        table.to_parquet(tmp, index=False)
        tmp.replace(OUT_PATH)
        logger.success(f"Saved {len(table):,} rows -> {OUT_PATH}")

    validation_report(corpus, table)


if __name__ == "__main__":
    main()
