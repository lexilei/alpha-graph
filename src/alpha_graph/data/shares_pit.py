"""Point-in-time shares outstanding from SEC XBRL company facts.

Cache: data/cache/shares_outstanding_pit.parquet, built by
scripts/fetch_shares_outstanding.py. One row per (ticker, end, filed):
`shares` were measured on `end` (cover-page / balance-sheet date) and
became publicly knowable on `filed` (the SEC filing date that disclosed
them). `concept` records the XBRL tag the row came from.

PIT rule implemented by :func:`shares_asof`: a share count is usable on
date D only if ``filed <= D`` (availability by FILING date, day
granularity — intraday acceptance-time gating is the job of
data/cache/filing_acceptance.parquet, not this table). Among the facts
available on D we take the latest measurement: max ``end``, ties broken
by latest ``filed`` so same-day corrections win and a late-filed
amendment that re-reports an old measurement date cannot roll the count
backwards.

Multi-class note: counts are company totals (classes summed at fetch
time), so sibling class tickers (GOOG/GOOGL, FOX/FOXA, NWS/NWSA) carry
the same series — multiplying each class's price by total shares
overstates a summed "market cap" across sibling rows. For filers whose
point counts are only tagged per class (META, HRL, ...), rows carry
concept us-gaap:WeightedAverageNumberOfSharesOutstandingBasic: a
within-period basic-EPS average standing in for a point count.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_graph.config import CACHE_DIR

SHARES_PIT_PATH = CACHE_DIR / "shares_outstanding_pit.parquet"

_FACT_COLS = ("end", "filed", "shares")


def load_shares_pit(path: Path | None = None) -> pd.DataFrame:
    """Load the shares-outstanding cache (all tickers, raw fact rows)."""
    return _load_cached(str(path or SHARES_PIT_PATH)).copy()


@lru_cache(maxsize=2)
def _load_cached(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df


def _state_series(facts: pd.DataFrame) -> pd.DataFrame:
    """Collapse one entity's fact rows into an as-of state table.

    - rows sharing (end, filed) are share classes split across facts:
      SUM them (fetch-time resolution already handles same-day
      re-filings; this is the defensive rule for raw frames);
    - sort by (filed, end) and keep only rows that weakly improve the
      running max of ``end`` — a row with a stale measurement date
      cannot supersede a newer one, while an equal ``end`` filed later
      (a correction) replaces the value;
    - one state per ``filed`` date (max ``end`` wins within a day).

    Returns a frame with columns (filed, shares) sorted by filed.
    """
    f = facts.loc[:, list(_FACT_COLS)].dropna()
    if f.empty:
        return f.iloc[:0].loc[:, ["filed", "shares"]]
    g = f.groupby(["end", "filed"], as_index=False)["shares"].sum()
    g = g.sort_values(["filed", "end"], kind="mergesort").reset_index(drop=True)
    g = g[g["end"] >= g["end"].cummax()]
    return g.drop_duplicates(subset="filed", keep="last").loc[:, ["filed", "shares"]]


def _facts_for(ticker_or_frame: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(ticker_or_frame, str):
        df = _load_cached(str(SHARES_PIT_PATH))
        return df[df["ticker"] == ticker_or_frame]
    frame = ticker_or_frame
    missing = [c for c in _FACT_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"facts frame lacks columns {missing}")
    if "ticker" in frame.columns and frame["ticker"].nunique() > 1:
        raise ValueError(
            "facts frame holds multiple tickers; pass a ticker string or a "
            "single-ticker frame"
        )
    return frame


def shares_asof(
    ticker_or_frame: str | pd.DataFrame,
    dates,
) -> float | pd.Series:
    """Latest share count known on each date (availability by FILED date).

    Parameters
    ----------
    ticker_or_frame : ticker string (looked up in the cache) or a facts
        DataFrame with columns end/filed/shares for a single entity.
    dates : scalar date-like, or a list-like of dates.

    Returns a float for a scalar date, else a pd.Series indexed by
    ``dates``. NaN where no fact had been filed yet (or the ticker has
    no facts at all).
    """
    states = _state_series(_facts_for(ticker_or_frame))

    scalar = np.ndim(dates) == 0 and not isinstance(dates, (list, tuple, set))
    dt = pd.DatetimeIndex(pd.to_datetime([dates] if scalar else list(dates)))

    if states.empty:
        out = np.full(len(dt), np.nan)
    else:
        idx = np.searchsorted(states["filed"].values, dt.values, side="right") - 1
        out = np.where(idx >= 0, states["shares"].values[np.maximum(idx, 0)], np.nan)
    if scalar:
        return float(out[0])
    return pd.Series(out, index=dt, name="shares")


def market_cap_panel(
    prices: pd.DataFrame,
    facts: pd.DataFrame | None = None,
) -> pd.Series:
    """Availability-correct market cap: shares_asof(ticker, date) x close.

    Parameters
    ----------
    prices : DataFrame with columns ticker, date, close (long format).
    facts : optional facts frame (defaults to the cache); mainly for tests.

    Returns a pd.Series aligned to ``prices.index`` (NaN where no share
    count was available yet). The shares side is point-in-time; the cap
    is only as honest as the closes you pass: use split-unadjusted
    closes for historically exact caps — with split/dividend-adjusted
    closes (e.g. market_data.parquet) a cap is understated by the split
    factor before a ticker's later splits.
    """
    missing = [c for c in ("ticker", "date", "close") if c not in prices.columns]
    if missing:
        raise ValueError(f"prices frame lacks columns {missing}")
    if facts is None:
        facts = _load_cached(str(SHARES_PIT_PATH))

    dates = pd.to_datetime(prices["date"]).values
    closes = prices["close"].values.astype(float)
    out = np.full(len(prices), np.nan)
    by_ticker = dict(iter(facts.groupby("ticker"))) if len(facts) else {}
    for ticker, pos in prices.groupby("ticker").indices.items():
        sub = by_ticker.get(ticker)
        if sub is None:
            continue
        shares = shares_asof(sub, dates[pos])
        out[pos] = shares.values * closes[pos]
    return pd.Series(out, index=prices.index, name="market_cap")
