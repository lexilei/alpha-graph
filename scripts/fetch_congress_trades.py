"""Fetch congressional STOCK Act PTR trades (both chambers) with per-trade
transaction AND disclosure dates -> data/cache/congress_trades.parquet.

DATA ACQUISITION ONLY for a future C20 congressional-trading factor: no factor,
no registry entry, no ledger rows here. PIT rule for any later use: a trade is
usable the day AFTER disclosure_date (the day the PTR became public).

Source survey (2026-07-14):
- Senate/House Stock Watcher (the classic free JSON dumps): DEAD. Both sites
  unreachable, both S3 buckets return AccessDenied, the GitHub data repo
  (timothycarambat/senate-stock-watcher-data) last committed 2021-03, and the
  jeremiak eFD mirror stopped 2024-01.
- kadoa-org/congress-trading-monitor (GitHub, MIT license): CHOSEN. Daily
  auto-refreshed static JSON parsed from the official House Clerk Financial
  Disclosure portal + Senate eFD (+ OGE 278-T for the executive branch, which
  this script drops). Per-trade transaction_date AND filing_date (= disclosure
  date), owner, amount range, doc_url back to the official PDF. ~64.6k rows
  2011 -> today; HEAD verified refreshed same-day at fetch time.
  Coverage, measured against the official House FD index (see cross-check in
  the report): e-filed House PTR filings are parsed at 98-99% for 2021+,
  ~79-93% 2018-2020, only 46-59% 2014-2017; paper filings (DocID not starting
  with "2"; 30-60% of filings in 2014-17 falling to ~10-15% by 2023+) are
  never parsed. Senate: e-filed only, pre-2015 paper skipped upstream, no
  official bulk index exists to audit against. So the usable era is
  effectively 2014+ (thin until ~2018).
- Official House Clerk yearly FD index zips (disclosures-clerk.house.gov,
  updated daily): filing-level metadata only (member, FilingType, FilingDate,
  DocID) — no transactions (those live in per-document PDFs). Fetched here as
  the coverage cross-check denominator.
- capitolgains (PyPI, MIT): a live scraper of both chambers, not a dataset —
  the DIY fallback if the kadoa repo dies.
- Quiver / Capitol Trades / FMP / Apify actors: paid or metered; excluded.

Normalization notes:
- ticker: upper, dots -> dashes (BRK.B -> BRK-B, matching the panel
  vocabulary), junk/CUSIP-like strings -> null. Mechanical only — historical
  renames (FB -> META etc.) are pit_universe.RENAME_MAP's job at factor time.
  Source quirk: kadoa's ticker is MOSTLY as-filed (115 FB rows) but its
  resolver occasionally emits the modern symbol (a 2020 "Facebook (FB)"
  exercise row carries META), so treat the symbol as approximately-as-filed
  and always run the rename map before joining the panel.
- asset classes: equity and option rows are kept (options flagged is_option;
  today's C16 lesson makes them composition-test material), everything else
  (bonds, munis, treasuries, funds, crypto, ...) is dropped. House [XX]
  schedule codes, Senate labels, OCR-cased "[sT]" tags in asset_name, and
  description keywords are all consulted; rows with a ticker but no
  classifiable type are kept as equity with asset_class_inferred=True.
- owner: SP/Spouse -> spouse, JT/Joint -> joint, DC/Child -> child,
  Self -> self, missing -> undisclosed (house blanks are NOT assumed self:
  parse gaps and blank-on-form are indistinguishable in the source).
- dedup: the same trade re-listed by a later (amended/overlapping) filing is
  collapsed to its EARLIEST disclosure_date — first public availability —
  and flagged is_refiled. Identical rows within one filing (repeated lots)
  are preserved via an occurrence counter. Explicit amendment markers do not
  exist in the source, so is_refiled is the amendment proxy.

Usage:
    python scripts/fetch_congress_trades.py              # fetch if stale, build, report
    python scripts/fetch_congress_trades.py --refresh    # force re-download
    python scripts/fetch_congress_trades.py --no-fetch   # offline: rebuild from raw
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alpha_graph.config import CACHE_DIR, DATA_DIR  # noqa: E402

KADOA_REPO = "https://github.com/kadoa-org/congress-trading-monitor"
KADOA_TARBALL = KADOA_REPO + "/archive/refs/heads/main.tar.gz"
KADOA_HEAD_API = "https://api.github.com/repos/kadoa-org/congress-trading-monitor/commits/main"
HOUSE_FD_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"

RAW_DIR = DATA_DIR / "raw" / "congress"
KADOA_DIR = RAW_DIR / "kadoa"
FD_DIR = RAW_DIR / "house_fd_index"
OUT_PATH = CACHE_DIR / "congress_trades.parquet"
PANEL_PATH = CACHE_DIR / "market_data.parquet"

FD_START_YEAR = 2012  # STOCK Act era
MAX_AGE_DAYS = 7.0
TIMEOUT = 60

# ---------------------------------------------------------------- asset classes
# House FD schedule codes (form instructions) + Senate eFD labels, upper-cased.
EQUITY_TYPES = {"ST", "STOCK", "RS", "PS", "NON-PUBLIC STOCK"}
OPTION_TYPES = {"OP", "STOCK OPTION", "SA"}  # SA = stock appreciation right
NON_EQUITY_TYPES = {
    "GS", "CS", "AB", "HN", "OI", "OL", "VA", "CT", "WU", "DB", "RP", "MF",
    "EF", "ET", "OT", "BA", "IH", "FA", "EIF", "5C", "5F", "5P", "4K",
    "MUNICIPAL SECURITY", "CORPORATE BOND", "GOVERNMENT SECURITY",
    "COMMODITIES/FUTURES CONTRACT", "CRYPTOCURRENCY", "OTHER",
    "OTHER SECURITIES",
}
_TAG_RE = re.compile(r"\[(\w{2})\]")
_OPTION_DESC_RE = re.compile(r"\b(puts?|calls?|options?)\b", re.IGNORECASE)
_NON_EQUITY_DESC_RE = re.compile(
    r"\b(treasur\w+|t-?bills?|municipal|bonds?|notes? due|cusip|money market"
    r"|mutual fund|etf|exchange[- ]traded|cryptocurrenc\w+|bitcoin|annuit\w+)\b",
    re.IGNORECASE,
)

_JUNK_TICKERS = {"", "-", "--", "N/A", "NA", "NONE", "NULL", "UNKNOWN", "#N/A"}

OWNER_MAP = {
    "SP": "spouse", "SPOUSE": "spouse",
    "JT": "joint", "JOINT": "joint",
    "DC": "child", "CHILD": "child", "DEPENDENT CHILD": "child",
    "SELF": "self",
}

COLUMNS = [
    "member", "member_id", "chamber", "party", "state",
    "ticker", "transaction_date", "notification_date", "disclosure_date",
    "disclosure_lag_days", "type", "type_raw", "amount_min", "amount_max",
    "owner", "owner_raw", "is_option", "asset_class_inferred",
    "asset_description", "asset_type_raw", "filing_id", "doc_url", "source",
    "is_refiled",
]


# ---------------------------------------------------------------- normalization
def normalize_ticker(t: object) -> str | None:
    """Upper-case, dots -> dashes (panel vocabulary), junk/CUSIP-ish -> None."""
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return None
    s = str(t).strip().upper()
    if s in _JUNK_TICKERS or " " in s:
        return None
    s = s.replace(".", "-")
    if re.search(r"\d{3,}", s):  # CUSIPs / bond identifiers (e.g. 91282CLC3)
        return None
    return s or None


def parse_amount_label(label: object) -> tuple[int | None, int | None]:
    """'$1,001 - $15,000' -> (1001, 15000); 'Over $50,000,000' -> (50000001, None)."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return (None, None)
    s = str(label)
    nums = [int(n.replace(",", "")) for n in re.findall(r"\$([\d,]+)", s)]
    if len(nums) >= 2:
        return (nums[0], nums[1])
    if len(nums) == 1:
        if re.search(r"over|\+", s, re.IGNORECASE):
            return (nums[0] + 1, None)
        return (nums[0], nums[0])
    return (None, None)


def classify_asset(asset_type: object, asset_name: object,
                   ticker: str | None) -> tuple[str, bool]:
    """-> (asset_class, inferred). asset_class in {'equity','option','non_equity',
    'unclassified'}; inferred=True when only the ticker's presence decided."""
    at = "" if asset_type is None or (isinstance(asset_type, float) and pd.isna(asset_type)) \
        else str(asset_type).strip().upper()
    name = "" if asset_name is None or (isinstance(asset_name, float) and pd.isna(asset_name)) \
        else str(asset_name)
    if at in OPTION_TYPES:
        return ("option", False)
    if at in EQUITY_TYPES:
        return ("equity", False)
    if at in NON_EQUITY_TYPES:
        return ("non_equity", False)
    m = _TAG_RE.search(name)  # e.g. "[ST]", OCR-cased "[sT]"
    if m:
        tag = m.group(1).upper()
        if tag in OPTION_TYPES:
            return ("option", False)
        if tag in EQUITY_TYPES:
            return ("equity", False)
        if tag in NON_EQUITY_TYPES:
            return ("non_equity", False)
    if _OPTION_DESC_RE.search(name):
        return ("option", False)
    if _NON_EQUITY_DESC_RE.search(name):
        return ("non_equity", False)
    if ticker:
        return ("equity", True)
    return ("unclassified", False)


def normalize_owner(o: object) -> str:
    if o is None or (isinstance(o, float) and pd.isna(o)):
        return "undisclosed"
    return OWNER_MAP.get(str(o).strip().upper(), "undisclosed")


def normalize_type(t: object) -> str:
    s = "" if t is None or (isinstance(t, float) and pd.isna(t)) else str(t).strip().upper()
    if s.startswith("PURCHASE"):
        return "purchase"
    if s.startswith("SALE"):
        return "sale"
    if s.startswith("EXCHANGE"):
        return "exchange"
    return "other"


def dedup_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the same trade re-listed by multiple filings to its earliest
    disclosure_date (is_refiled=True). Repeated identical lots within a single
    filing are distinct trades and are preserved via an occurrence counter."""
    df = df.copy()
    df["_desc"] = df["asset_description"].fillna("").str.strip().str.casefold()
    key = ["member_id", "transaction_date", "ticker", "_desc", "type",
           "amount_min", "amount_max", "owner"]
    df["_occ"] = df.groupby(["filing_id"] + key, dropna=False).cumcount()
    df = df.sort_values(["disclosure_date", "filing_id"], kind="stable")
    grp = df.groupby(key + ["_occ"], dropna=False)
    df["is_refiled"] = grp["filing_id"].transform("nunique") > 1
    df = grp.head(1)
    return df.drop(columns=["_desc", "_occ"])


def build_table(trades: pd.DataFrame, filer_map: dict[str, dict]) -> pd.DataFrame:
    """Kadoa raw trade rows + filer metadata -> normalized congress_trades table.

    Pure (no I/O): congressional PTR rows only, equity+option kept (options
    flagged), amounts as Int64, dates as datetime64, deduped across filings."""
    df = trades.copy()

    n0 = len(df)
    df = df[df["filing_type"] == "PTR"]  # drops OGE executive 278-T rows
    logger.info(f"rows: {n0} raw -> {len(df)} congressional PTR")

    fl = pd.DataFrame.from_dict(filer_map, orient="index")
    for col, src in [("member", "full_name"), ("chamber", "chamber"),
                     ("party", "party"), ("state", "state")]:
        df[col] = df["filer_id"].map(fl[src])
    df = df.rename(columns={"filer_id": "member_id", "source_id": "source",
                            "asset_name": "asset_description",
                            "asset_type": "asset_type_raw",
                            "transaction_type": "type_raw",
                            "owner": "owner_raw",
                            "filing_date": "disclosure_date"})

    for c in ("transaction_date", "notification_date", "disclosure_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    n_bad = int(df["transaction_date"].isna().sum() + df["disclosure_date"].isna().sum())
    if n_bad:
        logger.warning(f"dropping {n_bad} rows with unparseable transaction/disclosure date")
    df = df.dropna(subset=["transaction_date", "disclosure_date"])

    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["type"] = df["type_raw"].map(normalize_type)
    df["owner"] = df["owner_raw"].map(normalize_owner)

    # amounts: numeric fields are primary, label parse is the fallback
    lab = df["amount_range_label"].map(parse_amount_label)
    df["amount_min"] = df["amount_range_low"].fillna(lab.str[0]).astype("Int64")
    df["amount_max"] = df["amount_range_high"].fillna(lab.str[1]).astype("Int64")

    cls = [classify_asset(a, n, t) for a, n, t in
           zip(df["asset_type_raw"], df["asset_description"], df["ticker"])]
    df["asset_class"] = [c[0] for c in cls]
    df["asset_class_inferred"] = [c[1] for c in cls]
    drop_counts = df.loc[~df["asset_class"].isin(["equity", "option"]),
                         "asset_class"].value_counts().to_dict()
    logger.info(f"asset filter: keeping equity+option, dropping {drop_counts}")
    df = df[df["asset_class"].isin(["equity", "option"])]
    df["is_option"] = df["asset_class"] == "option"

    n1 = len(df)
    df = dedup_trades(df)
    logger.info(f"dedup: {n1} -> {len(df)} rows "
                f"({int(df['is_refiled'].sum())} kept rows had later re-filings)")

    df["disclosure_lag_days"] = (df["disclosure_date"] - df["transaction_date"]).dt.days
    neg = int((df["disclosure_lag_days"] < 0).sum())
    if neg:
        logger.warning(f"{neg} rows have disclosure BEFORE transaction (kept; check source)")

    return (df[COLUMNS]
            .sort_values(["disclosure_date", "member_id", "ticker"])
            .reset_index(drop=True))


# ---------------------------------------------------------------------- fetch
def _get(url: str, session: requests.Session) -> requests.Response:
    for attempt in (1, 2):
        try:
            r = session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            logger.warning(f"retrying {url}: {e}")
            time.sleep(2)
    raise RuntimeError("unreachable")


def fetch_kadoa(session: requests.Session) -> None:
    """Download the repo tarball, keep it as the raw original, extract
    public/data/ -> data/raw/congress/kadoa/, and record provenance."""
    logger.info(f"downloading {KADOA_TARBALL}")
    blob = _get(KADOA_TARBALL, session).content
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "kadoa_repo.tar.gz").write_bytes(blob)

    KADOA_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf.getmembers():
            parts = Path(m.name).parts  # strip "<repo>-main/public/data/"
            if len(parts) > 3 and parts[1:3] == ("public", "data") and m.isfile():
                rel = Path(*parts[3:])
                dest = KADOA_DIR / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                fobj = tf.extractfile(m)
                assert fobj is not None
                dest.write_bytes(fobj.read())
                n += 1
    logger.info(f"extracted {n} files -> {KADOA_DIR}")

    prov = {"source_repo": KADOA_REPO, "license": "MIT",
            "fetched_at": date.today().isoformat()}
    try:
        head = _get(KADOA_HEAD_API, session).json()
        prov["commit"] = head["sha"]
        prov["commit_date"] = head["commit"]["committer"]["date"]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not record HEAD commit: {e}")
    (KADOA_DIR / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))


def fetch_house_fd_index(session: requests.Session, refresh: bool) -> None:
    """Official House Clerk yearly FD index zips (coverage cross-check)."""
    FD_DIR.mkdir(parents=True, exist_ok=True)
    cur = date.today().year
    for year in range(FD_START_YEAR, cur + 1):
        dest = FD_DIR / f"{year}FD.zip"
        stale = (not dest.exists()
                 or (year == cur and (time.time() - dest.stat().st_mtime) / 86400 > MAX_AGE_DAYS)
                 or refresh)
        if not stale:
            continue
        try:
            dest.write_bytes(_get(HOUSE_FD_URL.format(year=year), session).content)
            logger.info(f"fetched {dest.name}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{dest.name}: fetch failed ({e}); cross-check will be partial")


# ----------------------------------------------------------------------- load
def load_kadoa() -> tuple[pd.DataFrame, dict[str, dict]]:
    rows: list[dict] = []
    filer_map: dict[str, dict] = {}
    files = sorted((KADOA_DIR / "filer").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no filer JSONs under {KADOA_DIR / 'filer'} — run without "
                                "--no-fetch first")
    for fp in files:
        d = json.loads(fp.read_text())
        filer_map[d["filer"]["id"]] = d["filer"]
        rows.extend(d["trades"])
    return pd.DataFrame(rows), filer_map


def load_house_fd_index() -> pd.DataFrame | None:
    """PTR filings (FilingType 'P') from the official index; None if missing."""
    frames = []
    for zp in sorted(FD_DIR.glob("*FD.zip")):
        try:
            with zipfile.ZipFile(zp) as z:
                name = [n for n in z.namelist() if n.endswith(".txt")][0]
                t = pd.read_csv(io.BytesIO(z.read(name)), sep="\t", dtype=str)
            t["index_year"] = int(zp.name[:4])
            frames.append(t)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{zp.name}: unreadable ({e})")
    if not frames:
        return None
    idx = pd.concat(frames, ignore_index=True)
    return idx[idx["FilingType"] == "P"].copy()


# --------------------------------------------------------------------- report
def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("  CONGRESSIONAL PTR TRADES — VALIDATION REPORT")
    print("=" * 72)
    yr = df["transaction_date"].dt.year
    print(f"\nrows: {len(df)}  members: {df['member_id'].nunique()}  "
          f"window: {df['transaction_date'].min().date()} -> {df['transaction_date'].max().date()}"
          f"  (options: {int(df['is_option'].sum())}, refiled: {int(df['is_refiled'].sum())})")

    print("\nrows and distinct members per transaction year:")
    per = df.groupby([yr, "chamber"]).size().unstack(fill_value=0)
    per["members"] = df.groupby(yr)["member_id"].nunique()
    print(per.to_string())

    lag = df["disclosure_lag_days"]
    print("\ndisclosure lag (disclosure_date - transaction_date), days:")
    print(f"  median {lag.median():.0f}   p90 {lag.quantile(.9):.0f}   max {lag.max()}   "
          f"min {lag.min()}")
    print(f"  >45d (statutory limit): {(lag > 45).sum()} rows "
          f"({100 * (lag > 45).mean():.1f}%)  — late filings are real and expected")

    if PANEL_PATH.exists():
        panel = set(pd.read_parquet(PANEL_PATH, columns=["ticker"])["ticker"].unique())
        hit = df["ticker"].isin(panel)
        print(f"\npanel mapping ({len(panel)} tickers): {hit.sum()}/{len(df)} rows "
              f"({100 * hit.mean():.1f}%) map directly; "
              f"{100 * df[~df['is_option']]['ticker'].isin(panel).mean():.1f}% of equity rows")
        un = df.loc[~hit & df["ticker"].notna(), "ticker"].value_counts().head(15)
        print("  top unmatched tickers (renames/non-S&P):",
              ", ".join(f"{t}({n})" for t, n in un.items()))
    else:
        print(f"\npanel mapping: SKIPPED ({PANEL_PATH} not found)")

    print("\ntop-20 most-traded tickers:")
    top = df["ticker"].value_counts().head(20)
    print("  " + ", ".join(f"{t}({n})" for t, n in top.items()))

    print("\nPELOSI CHECK — household trades of panel names, 2020-2024:")
    pel = df[df["member"].str.contains("Pelosi", case=False, na=False)
             & (df["transaction_date"] >= "2020-01-01")
             & (df["transaction_date"] <= "2024-12-31")]
    if PANEL_PATH.exists():
        panel = set(pd.read_parquet(PANEL_PATH, columns=["ticker"])["ticker"].unique())
        pel = pel[pel["ticker"].isin(panel)]
    cols = ["transaction_date", "disclosure_date", "disclosure_lag_days", "ticker",
            "type", "is_option", "amount_min", "amount_max", "owner"]
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(pel[cols].sort_values("transaction_date").to_string(index=False))
    print("=" * 72)


def coverage_crosscheck(trades: pd.DataFrame) -> None:
    """kadoa-parsed House PTR documents vs the official FD index, per year.

    Runs on the RAW kadoa rows (pre asset-filter/dedup) so it measures filing
    parse coverage, not what survived normalization."""
    p = load_house_fd_index()
    if p is None:
        print("\nhouse coverage cross-check: SKIPPED (no FD index zips)")
        return
    p["efiled"] = p["DocID"].str.startswith("2")
    h = trades[trades["source_id"] == "house_clerk"].copy()
    h["docid"] = h["doc_url"].str.extract(r"/(\d+)\.pdf", expand=False)
    h["fy"] = h["doc_url"].str.extract(r"ptr-pdfs/(\d{4})/", expand=False).astype(float)
    parsed = h.dropna(subset=["docid"]).drop_duplicates("docid").groupby("fy").size()
    tbl = pd.DataFrame({
        "index_all": p.groupby("index_year").size(),
        "index_efiled": p[p["efiled"]].groupby("index_year").size(),
        "kadoa_parsed": parsed,
    }).fillna(0).astype(int)
    tbl["parsed/efiled_%"] = (100 * tbl["kadoa_parsed"]
                              / tbl["index_efiled"].clip(lower=1)).round(1)
    print("\nhouse PTR FILING coverage vs official Clerk FD index"
          " (paper filings are never parsed):")
    print(tbl.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + normalize congressional PTR trades")
    ap.add_argument("--refresh", action="store_true", help="force re-download of raw sources")
    ap.add_argument("--no-fetch", action="store_true", help="offline: rebuild from existing raw")
    args = ap.parse_args()

    if not args.no_fetch:
        stats = KADOA_DIR / "stats.json"
        age = (time.time() - stats.stat().st_mtime) / 86400 if stats.exists() else None
        session = requests.Session()
        session.headers.update({"User-Agent": "alpha-graph research fetch"})
        if args.refresh or age is None or age > MAX_AGE_DAYS:
            fetch_kadoa(session)
        else:
            logger.info(f"kadoa raw is {age:.1f} days old (<{MAX_AGE_DAYS:.0f}); skipping "
                        "download (--refresh to force)")
        fetch_house_fd_index(session, args.refresh)

    trades, filer_map = load_kadoa()
    df = build_table(trades, filer_map)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT_PATH)
    logger.success(f"saved {len(df)} rows -> {OUT_PATH}")

    report(df)
    coverage_crosscheck(trades)


if __name__ == "__main__":
    main()
