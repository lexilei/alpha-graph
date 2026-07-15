"""Build data/cache/xbrl_facts.parquet: AS-FILED accession-level XBRL fundamentals.

Source: SEC Financial Statement Data Sets (quarterly ZIPs of sub.txt/num.txt),
https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets
(the old /dera/data/financial-statement-data-sets URL redirects there). Each
quarterly ZIP holds every submission ACCEPTED in that calendar quarter; a
submission appears in exactly one vintage, so values are frozen as originally
filed. That is the point of this table: downstream consumers (e.g. SUE) must
use the value as FIRST filed for a period, never post-hoc revised values.
This table stores one row per (accession, tag, ddate, qtrs, uom) so that
first-filed selection is possible downstream; it does NOT dedupe periods
itself — amendments and restated comparatives sit alongside originals.

Filters applied per quarter:
  sub.txt  form in {10-K, 10-Q, 10-K/A, 10-Q/A} AND cik in our corpus
           (authoritative map = distinct cik of data/cache/filing_acceptance.parquet)
  num.txt  adsh survived the sub filter AND tag in WHITELIST AND
           segments empty (consolidated, no dimensional breakdowns: 82% of
           whitelist-tag rows are segment/product/geography slices) AND
           coreg empty (parent entity, not co-registrant subsidiaries) AND
           value present AND uom not an obvious non-USD currency (any bare
           3-uppercase-letter code other than USD; 'shares' passes).

qtrs semantics (SEC readme, verified on data): number of quarters the value
covers — 0 = point-in-time instant (Assets, shares outstanding), 1 = single
quarter, 4 = annual. Annual EPS in a 10-K is qtrs=4, NOT qtrs=0.

Output columns:
    adsh          digits-only accession number (str)
    cik           10-digit zero-padded CIK (str)
    ticker        corpus ticker for the CIK (a cik with several tickers ->
                  first alphabetically; logged)
    form          10-K / 10-Q / 10-K/A / 10-Q/A
    is_amendment  form ends with /A
    period        fiscal period end date of the submission (datetime)
    filed         official filing date (datetime)
    accepted      EDGAR acceptance datetime as given by the dataset (ET)
    tag           XBRL tag (27-tag whitelist: EPS/revenue/income/BS/cash-flow/
                  current-items/gross-profit/shares)
    version       taxonomy version, e.g. us-gaap/2025
    ddate         period end date the VALUE refers to (datetime; a filing
                  also carries prior-period comparatives, kept as filed)
    qtrs          duration in quarters (0 = instant)
    uom           unit of measure (USD, shares)
    value         numeric value as filed

Sorted by (cik, filed, tag). ZIPs are deleted right after extraction of the
needed rows (raw re-download is cheap); disk is checked up front.

Usage:
    python scripts/fetch_xbrl_facts.py                # skip if cache <30 days old
    python scripts/fetch_xbrl_facts.py --refresh      # force rebuild
    python scripts/fetch_xbrl_facts.py --start-quarter 2010q3 --end-quarter 2026q1
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from loguru import logger

from alpha_graph.config import CACHE_DIR

OUT_PATH = CACHE_DIR / "xbrl_facts.parquet"
CORPUS_PATH = CACHE_DIR / "filing_acceptance.parquet"
TMP_DIR = CACHE_DIR / "xbrl_fsds_tmp"
LANDING_URLS = (
    "https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets",
    "https://www.sec.gov/dera/data/financial-statement-data-sets",  # legacy alias
)
USER_AGENT = "Lei Zhihan leizhihan0606@gmail.com"  # SEC fair-access identity
KEEP_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
WHITELIST = {
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "CashAndCashEquivalentsAtCarryingValue",
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    # --- 2026-07-15 extension: cash flow, current items, gross-profit inputs ---
    # Operating cash flow; the ...ContinuingOperations variant is the fallback
    # for filers that tag only that form.
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    # Current balance-sheet items (working-capital / accrual inputs).
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "InventoryNet",
    "ReceivablesNetCurrent",
    "AccountsReceivableNetCurrent",  # fallback for ReceivablesNetCurrent
    "PropertyPlantAndEquipmentNet",
    # Gross profit and its inputs; CostOfGoodsAndServicesSold is the fallback
    # for CostOfRevenue.
    "GrossProfit",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
}
DEFAULT_START = "2010q3"
REQUEST_INTERVAL = 1.0  # polite sequential fetching, ~1 request/second
MAX_AGE_DAYS = 30.0  # source updates quarterly
NUM_CHUNK_ROWS = 1_000_000  # num.txt reaches ~6M rows/quarter in recent vintages
MIN_FREE_GB = 30.0

_last_request_t = 0.0


# --------------------------------------------------------------------------- #
# Quarters
# --------------------------------------------------------------------------- #

def parse_quarter(q: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})q([1-4])", q.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(f"bad quarter {q!r}, expected e.g. 2010q3")
    return int(m.group(1)), int(m.group(2))


def quarter_range(start: tuple[int, int], end: tuple[int, int]) -> list[str]:
    out, (y, q) = [], start
    while (y, q) <= end:
        out.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def last_complete_quarter() -> tuple[int, int]:
    today = pd.Timestamp.today()
    y, q = today.year, (today.month - 1) // 3 + 1
    return (y - 1, 4) if q == 1 else (y, q - 1)


# --------------------------------------------------------------------------- #
# HTTP (rate-limited, retry once per file)
# --------------------------------------------------------------------------- #

def _pace() -> None:
    global _last_request_t
    wait = REQUEST_INTERVAL - (time.monotonic() - _last_request_t)
    if wait > 0:
        time.sleep(wait)
    _last_request_t = time.monotonic()


def discover_zip_urls(session: requests.Session) -> dict[str, str]:
    """Scrape the landing page: quarter key ('2010q3') -> absolute ZIP URL.
    Resolving from the page keeps us correct if the file path moves again."""
    for url in LANDING_URLS:
        _pace()
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"landing page {url}: {e}")
            continue
        hrefs = re.findall(r'href="([^"]*?/(\d{4}q[1-4])\.zip)"', resp.text)
        if hrefs:
            found = {q: urljoin(resp.url, href) for href, q in hrefs}
            logger.info(f"landing page lists {len(found)} quarterly ZIPs "
                        f"({min(found)} -> {max(found)})")
            return found
    raise RuntimeError("could not enumerate ZIP links from any landing page URL")


def download_zip(session: requests.Session, url: str, dest: Path) -> bool:
    """Stream a quarterly ZIP to dest. Retry once; False on definitive miss."""
    for attempt in (0, 1):
        _pace()
        try:
            with session.get(url, timeout=300, stream=True) as resp:
                if resp.status_code == 404:
                    logger.warning(f"{url}: HTTP 404"
                                   + ("; retrying once" if attempt == 0 else ""))
                    if attempt == 1:
                        return False
                    time.sleep(2)
                    continue
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            return True
        except requests.RequestException as e:
            if attempt == 1:
                logger.error(f"{url}: {e}; giving up on this quarter")
                return False
            logger.warning(f"{url}: {e}; retrying in 5s")
            time.sleep(5)
    return False


# --------------------------------------------------------------------------- #
# Corpus map
# --------------------------------------------------------------------------- #

def load_corpus_map() -> dict[int, str]:
    """cik (int) -> ticker from the acceptance table (first alphabetically)."""
    acc = pd.read_parquet(CORPUS_PATH, columns=["cik", "ticker"]).drop_duplicates()
    multi = acc.groupby("cik")["ticker"].nunique()
    for cik in multi.index[multi > 1]:
        ts = sorted(acc.loc[acc.cik == cik, "ticker"])
        logger.info(f"cik {cik} has {len(ts)} tickers {ts}; using {ts[0]}")
    cik_ticker = acc.groupby("cik")["ticker"].min()
    logger.info(f"corpus: {len(cik_ticker)} CIKs, {acc.ticker.nunique()} tickers")
    return {int(c): t for c, t in cik_ticker.items()}


# --------------------------------------------------------------------------- #
# Per-quarter extraction
# --------------------------------------------------------------------------- #

def _read_sub(zf: zipfile.ZipFile, corpus_ciks: set[int]) -> pd.DataFrame:
    with zf.open("sub.txt") as f:
        sub = pd.read_csv(
            f, sep="\t", dtype=str, quoting=3, low_memory=False,
            encoding="utf-8", encoding_errors="replace",
            usecols=["adsh", "cik", "form", "period", "filed", "accepted"],
        )
    keep = sub["form"].isin(KEEP_FORMS) & sub["cik"].astype(int).isin(corpus_ciks)
    return sub[keep]


def _read_num(zf: zipfile.ZipFile, adshs: set[str]) -> pd.DataFrame:
    """Chunked scan of num.txt: whitelist tags of surviving accessions,
    consolidated entity only (segments and coreg both empty)."""
    non_usd = re.compile(r"[A-Z]{3}")
    parts = []
    with zf.open("num.txt") as f:
        for ch in pd.read_csv(
            f, sep="\t", dtype=str, quoting=3, low_memory=False,
            encoding="utf-8", encoding_errors="replace",
            chunksize=NUM_CHUNK_ROWS,
            usecols=["adsh", "tag", "version", "ddate", "qtrs", "uom",
                     "segments", "coreg", "value"],
        ):
            m = (ch["adsh"].isin(adshs) & ch["tag"].isin(WHITELIST)
                 & ch["segments"].isna() & ch["coreg"].isna() & ch["value"].notna())
            m &= ~(ch["uom"].str.fullmatch(non_usd).fillna(False) & (ch["uom"] != "USD"))
            if m.any():
                parts.append(ch.loc[m, ["adsh", "tag", "version", "ddate",
                                        "qtrs", "uom", "value"]])
    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=["adsh", "tag", "version", "ddate",
                                       "qtrs", "uom", "value"]))


def process_zip(path: Path, corpus_ciks: set[int],
                cik_ticker: dict[int, str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        sub = _read_sub(zf, corpus_ciks)
        if sub.empty:
            return pd.DataFrame()
        num = _read_num(zf, set(sub["adsh"]))
    df = num.merge(sub, on="adsh", how="left")
    cik_int = df["cik"].astype(int)
    df["cik"] = cik_int.map("{:010d}".format)
    df["ticker"] = cik_int.map(cik_ticker)
    df["is_amendment"] = df["form"].str.endswith("/A")
    df["adsh"] = df["adsh"].str.replace("-", "", regex=False)
    for col in ("period", "filed", "ddate"):
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    df["accepted"] = pd.to_datetime(df["accepted"], errors="coerce")
    df["qtrs"] = pd.to_numeric(df["qtrs"]).astype("int16")
    df["value"] = pd.to_numeric(df["value"]).astype("float64")
    return df[["adsh", "cik", "ticker", "form", "is_amendment", "period",
               "filed", "accepted", "tag", "version", "ddate", "qtrs",
               "uom", "value"]]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build_table(quarters: list[str]) -> tuple[pd.DataFrame, list[str]]:
    free_gb = shutil.disk_usage(CACHE_DIR).free / 1e9
    logger.info(f"free disk: {free_gb:.0f} GB (ZIPs are deleted after "
                f"extraction regardless; <{MIN_FREE_GB:.0f} GB would still be safe)")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    urls = discover_zip_urls(session)

    cik_ticker = load_corpus_map()
    corpus_ciks = set(cik_ticker)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    gaps: list[str] = []
    for i, q in enumerate(quarters, 1):
        if q not in urls:
            logger.warning(f"{q}: not listed on the landing page; recording gap")
            gaps.append(q)
            continue
        dest = TMP_DIR / f"{q}.zip"
        try:
            if not download_zip(session, urls[q], dest):
                gaps.append(q)
                continue
            df = process_zip(dest, corpus_ciks, cik_ticker)
        finally:
            dest.unlink(missing_ok=True)  # process-then-delete, cap disk
        if len(df):
            frames.append(df)
        logger.info(f"[{i}/{len(quarters)}] {q}: {len(df):,} facts, "
                    f"{df['adsh'].nunique() if len(df) else 0} submissions")

    if not frames:
        raise RuntimeError("no facts extracted from any quarter")
    table = pd.concat(frames, ignore_index=True)
    key = ["adsh", "tag", "version", "ddate", "qtrs", "uom"]
    ndup = int(table.duplicated(key).sum())
    if ndup:
        logger.warning(f"dropping {ndup} duplicate fact rows (same {key})")
        table = table.drop_duplicates(key, keep="first")
    table = table.sort_values(["cik", "filed", "tag", "ddate", "qtrs"],
                              ignore_index=True)
    try:
        TMP_DIR.rmdir()
    except OSError:
        pass
    return table, gaps


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #

def validation_report(table: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("  XBRL FACTS — VALIDATION")
    print("=" * 64)

    n_corpus = pd.read_parquet(CORPUS_PATH, columns=["cik"]).cik.nunique()
    eps = table[table.tag.isin(["EarningsPerShareBasic", "EarningsPerShareDiluted"])]
    print(f"\ncoverage: {eps.cik.nunique()}/{n_corpus} corpus CIKs have >=1 EPS fact; "
          f"{table.cik.nunique()} CIKs overall; {len(table):,} rows")
    print("rows per filed-year:")
    for y, n in table.groupby(table.filed.dt.year).size().items():
        print(f"  {y}  {n:8,d}")

    # ---- AAPL diluted EPS, as filed (each 10-K's own fiscal year) ----------
    aapl = table[(table.cik == "0000320193") & (table.form == "10-K")
                 & (table.tag == "EarningsPerShareDiluted")]
    print("\nAAPL 10-K diluted EPS qtrs distribution (own-FY rows are qtrs=4; "
          f"all EPS rows: {aapl.qtrs.value_counts().to_dict()})")
    own = aapl[(aapl.ddate == aapl.period) & (aapl.qtrs == 4)]
    print("AAPL diluted EPS AS-FILED, one row per 10-K (own fiscal year):")
    for _, r in own.sort_values("filed").iterrows():
        print(f"  FY end {r.ddate.date()}  filed {r.filed.date()}  "
              f"adsh {r.adsh}  EPS {r.value:>7.2f}")
    print("  (as-filed values are NOT retro-adjusted across filings: the 7:1")
    print("   split of 2014-06 and 4:1 of 2020-08 appear as level breaks")
    print("   between adjacent 10-Ks)")
    prior = aapl[(aapl.qtrs == 4) & (aapl.ddate < aapl.period)]
    both = prior.merge(own, on=["ddate", "qtrs"], suffixes=("_later", "_orig"))
    breaks = both[(both.value_later - both.value_orig).abs() > 0.005]
    if len(breaks):
        r = breaks.sort_values("filed_later").iloc[0]
        print(f"  cross-filing check, same FY {r.ddate.date()}: original 10-K "
              f"filed {r.filed_orig.date()} says {r.value_orig}, the 10-K filed "
              f"{r.filed_later.date()} restates it (post-split) as {r.value_later}")

    # ---- revision example: same (cik, tag, ddate) two accessions, two values
    key = ["cik", "tag", "ddate", "qtrs", "uom"]
    dup = table[table.duplicated(key, keep=False)]
    g = dup.groupby(key).agg(nval=("value", "nunique"), namend=("is_amendment", "sum"))
    cand = g[(g.nval > 1) & (g.namend > 0)]
    if cand.empty:
        cand = g[g.nval > 1]
    print("\nrevision example (why first-filed selection matters):")
    if cand.empty:
        print("  none found")
    else:
        ex = table.set_index(key).loc[[cand.index[0]]].reset_index()
        ex = ex.sort_values("filed")
        r0 = ex.iloc[0]
        print(f"  {r0.ticker} (cik {r0.cik}) {r0.tag} for period ending "
              f"{r0.ddate.date()} (qtrs={r0.qtrs}):")
        for _, r in ex.iterrows():
            print(f"    {r.form:6s} filed {r.filed.date()}  adsh {r.adsh}  "
                  f"value {r.value:,.4f}")

    # ---- SUE readiness: consecutive quarterly diluted EPS ------------------
    tickers = sorted(table.ticker.dropna().unique())
    sample = tickers[:: max(1, len(tickers) // 20)][:20]
    epsq = table[(table.tag == "EarningsPerShareDiluted") & (table.qtrs == 1)]
    print(f"\nSUE readiness ({len(sample)} sampled tickers): longest run of "
          "consecutive fiscal quarters with a qtrs=1 diluted-EPS fact")
    ready = 0
    for t in sample:
        dd = epsq.loc[epsq.ticker == t, "ddate"].drop_duplicates().sort_values().tolist()
        run = best = 1 if dd else 0
        for a, b in zip(dd, dd[1:]):
            gap = (b - a).days
            run = run + 1 if 80 <= gap <= 100 else 1
            best = max(best, run)
        ok = best >= 12
        ready += ok
        print(f"  {t:6s} {len(dd):3d} quarter-ends, longest run {best:3d} "
              f"{'OK' if ok else '--'}")
    print(f"  => {ready}/{len(sample)} tickers have >=12 consecutive quarters")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build as-filed XBRL fundamentals from SEC Financial Statement Data Sets")
    p.add_argument("--refresh", action="store_true",
                   help=f"Force rebuild even if the parquet is <{MAX_AGE_DAYS:.0f} days old")
    p.add_argument("--start-quarter", type=parse_quarter, default=parse_quarter(DEFAULT_START),
                   metavar="YYYYqN", help=f"first quarterly vintage (default {DEFAULT_START})")
    p.add_argument("--end-quarter", type=parse_quarter, default=last_complete_quarter(),
                   metavar="YYYYqN", help="last quarterly vintage (default: last complete quarter)")
    args = p.parse_args()

    quarters = quarter_range(args.start_quarter, args.end_quarter)
    age_days = ((time.time() - OUT_PATH.stat().st_mtime) / 86400
                if OUT_PATH.exists() else None)
    if not args.refresh and age_days is not None and age_days < MAX_AGE_DAYS:
        logger.info(f"{OUT_PATH.name} is {age_days:.1f} days old (<{MAX_AGE_DAYS:.0f}); "
                    "skipping fetch (--refresh to force)")
        table = pd.read_parquet(OUT_PATH)
    else:
        logger.info(f"building {quarters[0]} -> {quarters[-1]} ({len(quarters)} quarters)")
        table, gaps = build_table(quarters)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_PATH.with_suffix(".parquet.tmp")
        table.to_parquet(tmp, index=False)
        tmp.replace(OUT_PATH)
        logger.success(f"Saved {len(table):,} rows -> {OUT_PATH}")
        if gaps:
            logger.warning(f"gap report: {len(gaps)} quarters missing: {gaps}")
        else:
            logger.info("gap report: none, all requested quarters processed")

    validation_report(table)


if __name__ == "__main__":
    main()
