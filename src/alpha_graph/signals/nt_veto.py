"""C18 `nt_filing_veto`: Form 12b-25 late-filing exclusion windows.

Registered 2026-07-13 (FACTORS.md C18); evaluation protocol pinned the same
day in reports/factor_preregistration.md. Construction implemented here:

- NT filing = any data/cache/nt_filings.parquet row (forms normalized to
  strip/upper at fetch; startswith "NT 10-K"/"NT 10-Q" covers the /A and
  transition variants). Amendments count as NT filings for the history rule
  but, arriving days after their parent, are blocked by it as triggers.
- Trigger ("first or rare"): an NT filing at date t opens a veto window only
  when the same panel ticker has NO other NT filing dated in [t - 3 years, t)
  (pd.DateOffset years; an NT exactly 3 years earlier still blocks, one filed
  3 years + 1 day earlier does not). Blocked filings still count as history
  for later ones. Same-day rows (e.g. NT 10-K and NT 10-Q filed together)
  collapse to one trigger: earliest acceptance_ts, then accession.
- Window: the 126 trading days starting at the first trading day >=
  filing_date + 1 calendar day (next-close availability), on the
  market_data.parquet calendar; clipped at the calendar end. Triggers whose
  availability date precedes the calendar start are dropped with a warning
  (their true window cannot be reconstructed on the truncated calendar);
  triggers past the calendar end get no window.
- The veto acts at REBALANCE dates only: a name inside a window at a
  month-end close is excluded from the book formed at that close; a window
  opening mid-month never removes an already-held name (`book_holdings`
  evaluates windows at forming dates only).

`ew_book_returns` is the plain-pandas equal-weight buy-and-hold book used by
the pinned C18 exclusion-delta look (scripts/c18_exclusion_delta.py) — it
deliberately does NOT touch the portfolio engine.

Cache: data/cache/nt_veto.parquet — ticker, veto_from, veto_to,
trigger_accession.

Usage:
    python -m alpha_graph.signals.nt_veto
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR

NT_PATH = CACHE_DIR / "nt_filings.parquet"
MARKET_PATH = CACHE_DIR / "market_data.parquet"
OUT_PATH = CACHE_DIR / "nt_veto.parquet"

LOOKBACK_YEARS = 3      # first-or-rare: no other NT in the prior 3 years
WINDOW_TDAYS = 126      # veto length in trading days

WINDOW_COLUMNS = ["ticker", "veto_from", "veto_to", "trigger_accession"]


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def first_or_rare(nt: pd.DataFrame, lookback_years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """Trigger rows of `nt`: no other NT filing by the same ticker dated in
    [t - lookback_years, t). Same-day rows collapse to one trigger (earliest
    acceptance_ts, then accession). Requires columns ticker, filing_date,
    accession (acceptance_ts optional, used only for same-day ordering)."""
    if not len(nt):
        return nt.head(0).copy()
    df = nt.copy()
    if "acceptance_ts" not in df.columns:
        df["acceptance_ts"] = pd.NaT
    df = df.sort_values(["ticker", "filing_date", "acceptance_ts", "accession"],
                        na_position="last")

    keep = []
    for _, g in df.groupby("ticker", sort=False):
        dates = g["filing_date"]
        for idx, t in dates.items():
            prior = dates[dates < t]
            if not (prior >= t - pd.DateOffset(years=lookback_years)).any():
                keep.append(idx)
    trig = df.loc[keep]
    return trig.drop_duplicates(["ticker", "filing_date"], keep="first")


def veto_windows(triggers: pd.DataFrame, calendar: Sequence[pd.Timestamp],
                 window_tdays: int = WINDOW_TDAYS) -> pd.DataFrame:
    """[ticker, veto_from, veto_to, trigger_accession] — window = the
    `window_tdays` trading days from the first calendar day >= filing_date +
    1 calendar day, clipped at the calendar end. Pre-calendar availability
    drops the trigger (warned); post-calendar availability yields no window."""
    cal = pd.DatetimeIndex(calendar).sort_values()
    rows = []
    for r in triggers.sort_values(["ticker", "filing_date"]).itertuples():
        avail = r.filing_date + pd.Timedelta(days=1)
        if avail < cal[0]:
            logger.warning(f"{r.ticker}: NT {r.filing_date.date()} available "
                           f"before the calendar start — window dropped")
            continue
        i0 = int(np.searchsorted(cal, avail, side="left"))
        if i0 >= len(cal):
            continue
        rows.append({
            "ticker": r.ticker,
            "veto_from": cal[i0],
            "veto_to": cal[min(i0 + window_tdays - 1, len(cal) - 1)],
            "trigger_accession": r.accession,
        })
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def build_veto(nt: pd.DataFrame, calendar: Sequence[pd.Timestamp]) -> pd.DataFrame:
    triggers = first_or_rare(nt)
    windows = veto_windows(triggers, calendar)
    logger.info(f"NT rows {len(nt)} -> triggers {len(triggers)} "
                f"(first-or-rare) -> windows {len(windows)} (in-calendar)")
    return windows


def excluded_at(windows: pd.DataFrame, date: pd.Timestamp) -> frozenset:
    """Tickers inside a veto window at `date`."""
    if not len(windows):
        return frozenset()
    m = (windows["veto_from"] <= date) & (date <= windows["veto_to"])
    return frozenset(windows.loc[m, "ticker"])


# --------------------------------------------------------------------------- #
# EW buy-and-hold book (plain pandas; used by the pinned C18 look and tests)
# --------------------------------------------------------------------------- #

def book_holdings(close_wide: pd.DataFrame, rebalances: Sequence[pd.Timestamp],
                  members_at: Callable[[pd.Timestamp], frozenset],
                  windows: pd.DataFrame | None = None,
                  ) -> dict[pd.Timestamp, list[str]]:
    """Holdings formed at each rebalance close: PIT members with a close
    price that day, minus (when `windows` is given) names inside a veto
    window AT THE REBALANCE DATE — the only dates the veto acts on."""
    out: dict[pd.Timestamp, list[str]] = {}
    for d in rebalances:
        members = members_at(d)
        if members is None:
            raise ValueError(f"no membership as of {d.date()} — rebalance list "
                             "must be restricted to covered dates")
        priced = close_wide.columns[close_wide.loc[d].notna()]
        names = members & set(priced)
        if windows is not None:
            names -= excluded_at(windows, d)
        out[d] = sorted(names)
    return out


def ew_book_returns(close_wide: pd.DataFrame, rebalances: Sequence[pd.Timestamp],
                    holdings: Mapping[pd.Timestamp, Sequence[str]],
                    end: pd.Timestamp) -> pd.Series:
    """Daily close-to-close returns of the book: equal weight at each
    rebalance close, buy-and-hold (drifting weights) to the next rebalance
    (the last book runs to `end`). Missing closes inside a holding month
    forward-fill (0% on halted days); each name must have a close at its
    forming date. Returns a Series indexed by trading day, first day of each
    segment excluded (it is the forming close)."""
    rebalances = list(rebalances)
    segs: list[pd.Series] = []
    for d0, d1 in zip(rebalances, rebalances[1:] + [end]):
        names = list(holdings[d0])
        if not names:
            raise ValueError(f"empty book at {d0.date()}")
        seg = close_wide.loc[d0:d1, names].ffill()
        if seg.iloc[0].isna().any():
            missing = seg.columns[seg.iloc[0].isna()].tolist()
            raise ValueError(f"no close at forming date {d0.date()}: {missing}")
        wealth = seg.div(seg.iloc[0]).mean(axis=1)
        segs.append(wealth.pct_change().iloc[1:])
    return pd.concat(segs)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> None:
    nt = pd.read_parquet(NT_PATH)
    calendar = pd.DatetimeIndex(
        pd.read_parquet(MARKET_PATH, columns=["date"])["date"].unique()).sort_values()
    logger.info(f"calendar: {len(calendar)} trading days "
                f"{calendar[0].date()} -> {calendar[-1].date()}")

    windows = build_veto(nt, calendar)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(OUT_PATH, index=False)
    logger.success(f"Saved {len(windows)} veto windows -> {OUT_PATH}")

    print("\n" + "=" * 64)
    print("  C18 NT VETO WINDOWS")
    print("=" * 64)
    if len(windows):
        print(f"\n{len(windows)} windows, {windows['ticker'].nunique()} tickers:")
        for r in windows.itertuples():
            print(f"  {r.ticker:6s} {r.veto_from.date()} -> {r.veto_to.date()} "
                  f"({r.trigger_accession})")
    else:
        print("\nno windows (no in-calendar first-or-rare NT triggers)")
    print("=" * 64)


if __name__ == "__main__":
    main()
