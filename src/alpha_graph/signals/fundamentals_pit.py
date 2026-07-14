"""B8/B9 PIT fundamentals numerators: as-filed book equity and trailing
4-quarter net income, dated by the FILING that completed each value.

Input: data/cache/xbrl_facts.parquet (SEC Financial Statement Data Sets —
every value frozen as originally filed). First-filed discipline throughout:
for each (cik, tag, ddate, qtrs) the EARLIEST filed accession's value
(sue_pead.first_filed; ties by accepted, then adsh), so a revision or a
later comparative never replaces the original number.

Book equity (B8 `book_to_market_pit` numerator):
- `StockholdersEquity` (parent tag), qtrs=0 instant rows; where the parent
  tag is absent for a (cik, ddate),
  `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`
  is the fallback for that filer-period. "Absent" is a property of the
  (cik, ddate) record, not of a single filing: for ~1.8% of dual-tag
  periods (measured at build, e.g. YUM fiscal-year ends) the parent value
  is first filed only as a LATER filing's comparative (median 363d after
  the NCI variant) — those periods carry the parent value with its late
  availability, where the state collapse below usually supersedes them
  with a newer period the same day (84.5% of them have identical values
  anyway). Staleness, never lookahead.
- Negative book values are kept (real).
- avail_date = the filed date of the accession carrying the first-filed
  value.

Net income TTM (B9 `earnings_yield_pit` numerator):
- `NetIncomeLoss` qtrs=1 rows; where a filer lacks a standalone Q4,
  Q4 = annual (qtrs=4, the 10-K's own period) minus that fiscal year's
  three qtrs=1 values — sue_pead.quarterly_eps reused verbatim (same
  discipline as C17, incl. the 45-330d fiscal-window gate and the ANNUAL
  filing's availability for derived rows).
- TTM = sum of 4 consecutive quarters; the window must span at most
  TTM_MAX_SPAN_DAYS from oldest to newest quarter end (three 13-week gaps
  ~= 273d; a missing quarter pushes the span to ~364d), else NaN.
- avail_date = the max of the 4 constituent filings' filed dates (the
  filing that completes the trailing year).

Both event streams are collapsed to as-of STATES before saving (the
shares_pit._state_series discipline): rows sorted by avail_date must weakly
improve the running max of period_end — a delinquent old-period filing can
never roll the latest-known value backwards — and one row per
(ticker, avail_date) keeps the latest period. The saved table is therefore
directly usable in a backward merge_asof on avail_date; the availability
lag (v0: t+1) is applied at the panel merge like every other filing-dated
input, and the division by PIT market cap at t happens inside
build_feature_panel (B7's cap; see ml_combiner.py).

Cache: data/cache/fundamentals_pit.parquet — ticker, avail_date,
book_equity, net_income_ttm (one column NaN where only the other's event
fired that day).

Usage:
    python -m alpha_graph.signals.fundamentals_pit
"""

from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR
from alpha_graph.signals.sue_pead import first_filed, quarterly_eps

TAG_EQUITY = "StockholdersEquity"
TAG_EQUITY_NCI = ("StockholdersEquityIncludingPortionAttributableTo"
                  "NoncontrollingInterest")
TAG_NI = "NetIncomeLoss"
MONEY_UOM = "USD"
TTM_QUARTERS = 4              # trailing quarters in the B9 numerator
TTM_MAX_SPAN_DAYS = 320       # max ddate span of the 4 quarters (missing
                              # quarter => ~364d => TTM undefined)

CACHE_NAME = "fundamentals_pit.parquet"


# --------------------------------------------------------------------------- #
# Event streams
# --------------------------------------------------------------------------- #

def book_equity_events(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, ddate): first-filed parent stockholders' equity,
    else the including-NCI fallback where the parent tag is absent for that
    filer-period (qtrs=0 instant rows only).

    Returns cik, ticker, period_end, book_equity, avail_date (= the filed
    date of the accession carrying the value). Negative equity kept.
    """
    inst = facts[facts["qtrs"] == 0]
    par = first_filed(inst, TAG_EQUITY, MONEY_UOM)
    nci = first_filed(inst, TAG_EQUITY_NCI, MONEY_UOM)
    fb = nci.merge(par[["cik", "ddate"]].drop_duplicates(),
                   on=["cik", "ddate"], how="left", indicator=True)
    fb = fb[fb["_merge"] == "left_only"].drop(columns="_merge")
    logger.info(f"book equity: {len(par)} parent-tag periods, "
                f"{len(fb)} NCI-fallback periods")
    out = pd.concat([par, fb], ignore_index=True)
    out = out[["cik", "ticker", "ddate", "value", "filed"]].rename(
        columns={"ddate": "period_end", "value": "book_equity",
                 "filed": "avail_date"})
    return out.sort_values(["cik", "period_end"],
                           kind="mergesort").reset_index(drop=True)


def net_income_ttm_events(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, quarter) with a defined trailing-4-quarter NI sum.

    Quarterly rows come from sue_pead.quarterly_eps (first-filed qtrs=1
    values plus derived Q4 = annual - 3 quarters where no standalone Q4;
    derived rows carry the ANNUAL filing's filed date). TTM is NaN when the
    4 quarters span more than TTM_MAX_SPAN_DAYS (a quarter is missing from
    the record) or any constituent value is missing.

    Returns cik, ticker, period_end, net_income_ttm, avail_date (= max of
    the 4 constituent filings' filed dates).
    """
    first = first_filed(facts, TAG_NI, MONEY_UOM)
    q = quarterly_eps(first).rename(columns={"eps": "ni"})
    if q.empty:
        return pd.DataFrame({"cik": pd.Series(dtype=str),
                             "ticker": pd.Series(dtype=str),
                             "period_end": pd.Series(dtype="datetime64[ns]"),
                             "net_income_ttm": pd.Series(dtype=float),
                             "avail_date": pd.Series(dtype="datetime64[ns]")})
    parts = []
    for _, g in q.groupby("cik", sort=False):
        g = g.sort_values("ddate").reset_index(drop=True)
        g["net_income_ttm"] = g["ni"].rolling(TTM_QUARTERS).sum()
        span = (g["ddate"] - g["ddate"].shift(TTM_QUARTERS - 1)).dt.days
        g["net_income_ttm"] = g["net_income_ttm"].where(
            span <= TTM_MAX_SPAN_DAYS)
        filed = pd.concat([g["stmt_filed"].shift(k)
                           for k in range(TTM_QUARTERS)], axis=1)
        g["avail_date"] = filed.max(axis=1, skipna=False)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["net_income_ttm", "avail_date"])
    n_q = len(q)
    logger.info(f"net income TTM: {len(out)} defined windows from {n_q} "
                f"quarter rows ({q['derived_q4'].mean() * 100:.1f}% derived Q4)")
    out = out[["cik", "ticker", "ddate", "net_income_ttm", "avail_date"]]
    out = out.rename(columns={"ddate": "period_end"})
    return out.sort_values(["cik", "period_end"],
                           kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# As-of state collapse (the shares_pit discipline)
# --------------------------------------------------------------------------- #

def as_of_states(events: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Collapse an event stream to as-of states per ticker.

    Sort by (avail_date, period_end) and keep only rows that weakly improve
    the running max of period_end — a late-filed OLD period can never roll
    the latest-known value backwards — then one row per (ticker,
    avail_date), the latest period_end winning. The result is directly
    consumable by a backward merge_asof on avail_date.
    """
    e = events.dropna(subset=[value_col, "avail_date"])
    parts = []
    for _, g in e.groupby("ticker", sort=True):
        g = g.sort_values(["avail_date", "period_end"], kind="mergesort")
        g = g[g["period_end"] >= g["period_end"].cummax()]
        parts.append(g.drop_duplicates(subset="avail_date", keep="last"))
    if not parts:
        return events.iloc[:0].loc[:, ["ticker", "avail_date", value_col]]
    out = pd.concat(parts, ignore_index=True)
    return out.loc[:, ["ticker", "avail_date", value_col]]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def compute(facts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the fundamentals cache. `facts` is injectable for tests;
    the default reads xbrl_facts.parquet, and only real runs write the
    cache."""
    inject = facts is not None
    if facts is None:
        facts = pd.read_parquet(CACHE_DIR / "xbrl_facts.parquet")

    book = as_of_states(book_equity_events(facts), "book_equity")
    ttm = as_of_states(net_income_ttm_events(facts), "net_income_ttm")
    df = book.merge(ttm, on=["ticker", "avail_date"], how="outer")
    df = df.sort_values(["ticker", "avail_date"],
                        kind="mergesort").reset_index(drop=True)

    logger.info(
        f"fundamentals_pit: {len(df)} state rows, "
        f"{df['ticker'].nunique()} tickers; "
        f"book_equity on {df['book_equity'].notna().sum()}, "
        f"net_income_ttm on {df['net_income_ttm'].notna().sum()}; "
        f"negative book on {(df['book_equity'] < 0).sum()} rows"
    )
    if not inject:
        path = CACHE_DIR / CACHE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
    return df


def main():
    p = argparse.ArgumentParser(
        description="Build the B8/B9 fundamentals_pit cache")
    p.parse_args()
    df = compute()
    if not df.empty:
        print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
