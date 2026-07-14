"""Fetch Ken French daily factors -> data/cache/ff5_mom_daily.parquet.

Downloads the CSV zip endpoints of the Dartmouth data library:
  - F-F Research Data 5 Factors 2x3 (daily)   -> mkt_rf, smb, hml, rmw, cma, rf
  - F-F Momentum Factor (daily)               -> mom
parses them, converts percent -> decimal, and inner-joins on date. The joined
file starts 1963-07-01 (FF5 start; momentum alone reaches back to 1926) and
ends ~2 months behind today — the library updates on CRSP's schedule, which is
fine for attribution work.

Format hazards this parser guards against (verified on the 2026-06 vintage):
  - preamble length differs per file (4 vs 14 lines) and drifts between
    releases -> data rows are located by the ^YYYYMMDD, regex; the header is
    the line immediately above the first data row;
  - the momentum file carries a trailing comma on every line (header ",Mom,")
    -> unnamed/empty columns are dropped after parsing;
  - both files end with a blank line + copyright footer -> the data block
    stops at the first non-matching line, and a SECOND date block anywhere
    later (the monthly files append annual tables) raises instead of guessing;
  - documented missing codes -99.99 / -999 -> NaN, dropped with a count;
  - renamed/reordered columns raise with the found-vs-expected sets.

Usage:
  .venv/bin/python scripts/fetch_ff_factors.py            # reuse parquet if present
  .venv/bin/python scripts/fetch_ff_factors.py --refresh  # force re-download
"""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from loguru import logger

from alpha_graph.eval.ff_attribution import FF5_MOM_PARQUET

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
SOURCES = {
    "ff5": f"{BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": f"{BASE}/F-F_Momentum_Factor_daily_CSV.zip",
}
EXPECTED_COLS = {
    "ff5": {"mkt_rf", "smb", "hml", "rmw", "cma", "rf"},
    "mom": {"mom"},
}
COLUMN_ORDER = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
MISSING_CODES = (-99.99, -999.0)
DATA_ROW = re.compile(r"^\s*\d{8}\s*,")
USER_AGENT = "alpha-graph factor fetch (research; leizhihan0606@gmail.com)"


# --------------------------------------------------------------------------- #
# Download + parse
# --------------------------------------------------------------------------- #

def download_csv_text(url: str) -> str:
    """Fetch a library zip and return its single CSV member as text."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{url}: expected exactly one CSV in the zip, "
                             f"got {zf.namelist()}")
        return zf.read(members[0]).decode("latin-1")


def parse_french_daily(text: str, expected: set[str]) -> pd.DataFrame:
    """Parse one French daily CSV into a decimal-return frame.

    Locates data rows by the YYYYMMDD date regex (preamble length is not
    stable across releases), reads the single contiguous block, and raises on
    any later date block or on unexpected column names.
    """
    lines = text.splitlines()
    first = next((i for i, ln in enumerate(lines) if DATA_ROW.match(ln)), None)
    if first is None or first == 0:
        raise ValueError("no 'YYYYMMDD,' data rows found below a header — "
                         "library format changed?")
    block = []
    for ln in lines[first:]:
        if not DATA_ROW.match(ln):
            break
        block.append(ln)
    if any(DATA_ROW.match(ln) for ln in lines[first + len(block):]):
        raise ValueError("multiple date blocks in one file (monthly-style "
                         "annual appendix?) — refusing to guess which to keep")

    df = pd.read_csv(io.StringIO("\n".join([lines[first - 1]] + block)))
    df.columns = [str(c).strip() for c in df.columns]
    dates = pd.to_datetime(df.iloc[:, 0].astype(str).str.strip(), format="%Y%m%d")
    df = df.iloc[:, 1:].set_index(pd.DatetimeIndex(dates, name="date"))
    # trailing commas in the momentum file -> unnamed empty columns; drop them
    keep = [c for c in df.columns if c and not c.lower().startswith("unnamed")]
    df = df[keep]
    df.columns = [c.lower().replace("-", "_") for c in df.columns]
    if set(df.columns) != expected:
        raise ValueError(f"unexpected columns {sorted(df.columns)} "
                         f"(expected {sorted(expected)}) — library renamed them?")

    df = df.apply(pd.to_numeric, errors="raise")
    df = df.mask(df.isin(MISSING_CODES)) / 100.0  # percent -> decimal
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate dates in factor file")
    return df


def build_joined_frame() -> pd.DataFrame:
    frames = {}
    for key, url in SOURCES.items():
        logger.info(f"downloading {key}: {url}")
        frames[key] = parse_french_daily(download_csv_text(url), EXPECTED_COLS[key])
        logger.info(f"  {key}: {len(frames[key]):,} days, "
                    f"{frames[key].index[0].date()} -> {frames[key].index[-1].date()}")
    merged = frames["ff5"].join(frames["mom"], how="inner")[COLUMN_ORDER]
    n_before = len(merged)
    merged = merged.dropna()
    if n_before - len(merged):
        logger.warning(f"dropped {n_before - len(merged)} days with missing "
                       f"codes after the join")
    return merged


# --------------------------------------------------------------------------- #
# Coverage report + CLI
# --------------------------------------------------------------------------- #

def report_coverage(df: pd.DataFrame) -> None:
    start, end = df.index[0], df.index[-1]
    print(f"coverage : {start.date()} -> {end.date()}  ({len(df):,} trading days)")
    print(f"columns  : {list(df.columns)}")
    print(f"last row : {df.iloc[-1].to_dict()}")
    if start > pd.Timestamp("2011-01-01"):
        logger.warning(f"coverage starts {start.date()} — expected well before "
                       f"2011; check the parse")
    age = (pd.Timestamp.today().normalize() - end).days
    if age > 120:
        logger.warning(f"last day is {age}d old — beyond the library's usual "
                       f"~2-month publication lag; source may be stale")
    else:
        print(f"note     : last day is {age}d old — consistent with the "
              f"library's ~2-month publication lag (fine)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch Ken French FF5 + momentum daily factors to parquet")
    p.add_argument("--out", default=str(FF5_MOM_PARQUET),
                   help="output parquet path (default: data/cache/ff5_mom_daily.parquet)")
    p.add_argument("--refresh", action="store_true",
                   help="re-download even if the parquet already exists")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and not args.refresh:
        logger.info(f"{out} exists — reporting cached coverage (--refresh to re-download)")
        report_coverage(pd.read_parquet(out))
        return

    merged = build_joined_frame()
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out)
    logger.info(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    report_coverage(pd.read_parquet(out))  # report what actually round-trips


if __name__ == "__main__":
    main()
