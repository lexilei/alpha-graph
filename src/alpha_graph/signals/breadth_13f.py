"""C35 `breadth_13f` — Chen-Hong-Stein (2002) breadth of ownership.

Registered 2026-07-21 (ownership-flow member 2; construction and sign
pinned in reports/factor_preregistration.md before computation):

breadth(ticker, q) = # distinct 13F filers (CIKs) holding the ticker,
counted from each cik's FIRST-FILED 13F-HR for the period (amendments
/A excluded; late filings — FILING_DATE > period_end + 46 calendar
days, the statutory deadline — excluded: invisible at availability).
Position rows count only if SSHPRNAMTTYPE == "SH", PUTCALL is null,
and SSHPRNAMT > 0. Signal = breadth_q − breadth_{q−1} (plain Δ, OSAP-
faithful; the judge's within-month rank-z absorbs cross-time drift).

Partial-period guard: a period whose selected-filing count is < 50% of
the median of its ±2 neighbors is dropped as base AND current (kills
the 2013Q2 SEC-dataset stub). Availability = period_end + 46cd,
"filing" merge (v0 t+1). Sign hypothesis: POSITIVE.

Build: `.venv/bin/python -m alpha_graph.signals.breadth_13f`
→ data/cache/breadth_13f.parquet (ticker, avail_date, breadth_13f).
"""
from __future__ import annotations

import glob
import logging

import pandas as pd

from alpha_graph.config import CACHE_DIR

logger = logging.getLogger(__name__)

DEADLINE_DAYS = 46
GUARD_FRAC = 0.5
OUT_NAME = "breadth_13f.parquet"
USECOLS = ["ACCESSION_NUMBER", "CIK", "FILING_DATE", "SUBMISSIONTYPE",
           "PERIODOFREPORT", "ticker", "SSHPRNAMT", "SSHPRNAMTTYPE",
           "PUTCALL"]


def _reduce_file(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(accessions, holder-pairs) from one period file, filters applied."""
    df = df[df["SUBMISSIONTYPE"] == "13F-HR"]
    acc = (
        df[["ACCESSION_NUMBER", "CIK", "FILING_DATE", "PERIODOFREPORT"]]
        .drop_duplicates("ACCESSION_NUMBER")
    )
    pos = df[
        (df["SSHPRNAMTTYPE"] == "SH")
        & (df["PUTCALL"].isna())
        & (pd.to_numeric(df["SSHPRNAMT"], errors="coerce") > 0)
    ]
    pairs = pos[["ACCESSION_NUMBER", "ticker"]].drop_duplicates()
    return acc, pairs


def build_breadth(files: list[str]) -> pd.DataFrame:
    accs, pairs = [], []
    for f in files:
        df = pd.read_parquet(f, columns=USECOLS)
        a, p = _reduce_file(df)
        accs.append(a)
        pairs.append(p)
    acc = pd.concat(accs, ignore_index=True).drop_duplicates("ACCESSION_NUMBER")
    acc["period_end"] = pd.to_datetime(acc["PERIODOFREPORT"], format="%d-%b-%Y")
    acc["filing_date"] = pd.to_datetime(acc["FILING_DATE"], format="%d-%b-%Y")
    deadline = acc["period_end"] + pd.Timedelta(days=DEADLINE_DAYS)
    acc = acc[acc["filing_date"] <= deadline]
    # first-filed 13F-HR per (cik, period)
    acc = (
        acc.sort_values(["filing_date", "ACCESSION_NUMBER"], kind="mergesort")
        .drop_duplicates(["CIK", "period_end"], keep="first")
    )
    # partial-period guard on selected-filing counts
    counts = acc.groupby("period_end")["ACCESSION_NUMBER"].nunique().sort_index()
    med = counts.rolling(5, center=True, min_periods=2).median()
    ok_periods = counts[counts >= GUARD_FRAC * med].index
    dropped = sorted(set(counts.index) - set(ok_periods))
    if dropped:
        logger.info("partial-period guard dropped: %s",
                    [str(d.date()) for d in dropped])
    acc = acc[acc["period_end"].isin(ok_periods)]

    pair = pd.concat(pairs, ignore_index=True).drop_duplicates()
    held = pair.merge(
        acc[["ACCESSION_NUMBER", "CIK", "period_end"]],
        on="ACCESSION_NUMBER", how="inner",
    )
    breadth = (
        held.groupby(["ticker", "period_end"])["CIK"].nunique().rename("breadth")
        .reset_index()
        .sort_values(["ticker", "period_end"])
    )
    # Δ vs the immediately prior GUARD-PASSING quarter (~91d apart)
    prev = breadth.groupby("ticker")[["period_end", "breadth"]].shift(1)
    gap = (breadth["period_end"] - prev["period_end"]).dt.days
    delta = breadth["breadth"] - prev["breadth"]
    out = pd.DataFrame(
        {
            "ticker": breadth["ticker"],
            "avail_date": breadth["period_end"] + pd.Timedelta(days=DEADLINE_DAYS),
            "breadth_13f": delta.where((gap >= 80) & (gap <= 100)),
        }
    ).dropna()
    return out.sort_values(["ticker", "avail_date"]).reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = sorted(glob.glob(str(CACHE_DIR / "13f" / "*.parquet")))
    res = build_breadth(files)
    path = CACHE_DIR / OUT_NAME
    res.to_parquet(path, index=False)
    logger.info(
        "breadth_13f: %d rows / %d tickers, %s -> %s; |delta| median %.0f",
        len(res), res["ticker"].nunique(),
        res["avail_date"].min().date(), res["avail_date"].max().date(),
        res["breadth_13f"].abs().median(),
    )


if __name__ == "__main__":
    main()
