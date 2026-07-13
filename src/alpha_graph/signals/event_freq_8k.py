"""8-K event-frequency factors (C11 family).

The predictive content of 8-K filings in this universe is in the *abnormal
frequency* of filing activity, not in item content or tone: a spike in a firm's
8-K count relative to its own trailing baseline predicts lower forward returns
(the market under-reacts to clustered disclosure). Item-type and sentiment
variants were tested and rejected — see FACTORS.md / the pre-registration log.

Factor `evt8k_freq_z` (C11): z-score of the current calendar month's 8-K count
vs the firm's trailing 24-month count distribution. PIT (uses only months ≤ t).
NaN for tickers with no 8-K corpus (missing data, not "zero events").

Factor `evt8k_freq_z_d` (C15): the daily-grid variant. NOT a port of C11 — a
NEW registry entry, because three definitional pieces change at once:
  1. grid: every trading day t instead of calendar month-ends;
  2. scored window: the trailing 21-trading-day count instead of the
     calendar-month count (no weekend/short-month artifacts);
  3. baseline: mean/std of that same statistic over the trailing 504 trading
     days LAGGED by one full window, so the scored window never overlaps its
     own null (C11's rolling 24m baseline includes the scored month).
Rows are stamped with the computation date t (counts filings through t);
availability lag is applied at the panel merge (plan Item 2, single owner).

Usage:
    python -m alpha_graph.signals.event_freq_8k [--daily]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR

BASELINE_MONTHS = 24     # trailing window for the abnormal-frequency z-score
MIN_MONTHS = 6           # need this many trailing months before scoring

DAILY_WINDOW = 21        # C15 scored window, trading days
DAILY_BASELINE = 504     # C15 baseline lookback, trading days (~24 months)
DAILY_MIN_OBS = 126      # min baseline observations before scoring (~6 months)


def _filing_dates() -> pd.DataFrame:
    """One row per 8-K filing: ticker, filing date (day resolution)."""
    rows = []
    for d in FILINGS_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        for f in d.glob("8-K_*.json"):
            parts = f.name.split("_")
            if len(parts) >= 2:
                rows.append((d.name, parts[1]))
    df = pd.DataFrame(rows, columns=["ticker", "fdate"])
    df["fdate"] = pd.to_datetime(df["fdate"], errors="coerce")
    return df.dropna(subset=["fdate"])


def _filing_months() -> pd.DataFrame:
    """One row per 8-K filing: ticker, filing month (Period[M])."""
    df = _filing_dates()
    df["month"] = df["fdate"].dt.to_period("M")
    return df


def compute_all(start: str = "2011-01", end: str | None = None) -> pd.DataFrame:
    ev = _filing_months()
    if ev.empty:
        logger.warning("No 8-K filings found under FILINGS_DIR.")
        return pd.DataFrame()

    covered = sorted(ev["ticker"].unique())
    end = end or str(ev["month"].max())
    grid = pd.period_range(start, end, freq="M")
    counts = ev.groupby(["ticker", "month"]).size().rename("n8k")

    out = []
    for t in covered:
        s = counts.loc[t].reindex(grid, fill_value=0) if t in counts.index.get_level_values(0) \
            else pd.Series(0, index=grid)
        mu = s.rolling(BASELINE_MONTHS, min_periods=MIN_MONTHS).mean()
        sd = s.rolling(BASELINE_MONTHS, min_periods=MIN_MONTHS).std()
        z = (s - mu) / sd.replace(0, np.nan)
        out.append(pd.DataFrame({
            "ticker": t,
            "date": grid.to_timestamp(how="end").normalize(),
            "evt8k_freq_z": z.values,
            "evt8k_count_1m": s.values,
        }))
    df = pd.concat(out, ignore_index=True)
    df = df.dropna(subset=["evt8k_freq_z"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    path = CACHE_DIR / "event_freq_8k.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(
        f"Saved {len(df)} rows ({df['ticker'].nunique()} tickers with 8-K corpus, "
        f"{df['date'].min().date()}→{df['date'].max().date()}) to {path}"
    )
    return df


def trading_calendar() -> pd.DatetimeIndex:
    """Trading days = the price panel's own date vocabulary."""
    market = pd.read_parquet(CACHE_DIR / "market_data.parquet", columns=["date"])
    return pd.DatetimeIndex(np.sort(market["date"].unique()))


def compute_daily(
    start: str = "2011-01-03",
    end: str | None = None,
    events: pd.DataFrame | None = None,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """C15 `evt8k_freq_z_d` on every trading day (see module docstring).

    events/calendar are injectable for tests; defaults read the filing corpus
    and the price panel. Weekend/holiday filings count toward the next trading
    close (searchsorted left); filings outside the calendar are dropped.
    """
    ev = _filing_dates() if events is None else events.copy()
    if ev.empty:
        logger.warning("No 8-K filings found under FILINGS_DIR.")
        return pd.DataFrame()
    cal = trading_calendar() if calendar is None else pd.DatetimeIndex(calendar)
    cal = cal[cal >= pd.Timestamp(start)]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end)]

    ev["fdate"] = pd.to_datetime(ev["fdate"])
    ev = ev[ev["fdate"] >= cal[0]]           # pre-window filings can't reach any window
    idx = cal.searchsorted(ev["fdate"].to_numpy(), side="left")
    keep = idx < len(cal)
    ev = ev[keep].copy()
    if ev.empty:
        logger.warning("No 8-K filings inside the calendar window.")
        return pd.DataFrame()
    ev["tday"] = cal[idx[keep]]

    counts = ev.groupby(["ticker", "tday"]).size().rename("n")
    out = []
    for t in sorted(ev["ticker"].unique()):
        s = counts.loc[t].reindex(cal, fill_value=0).astype(float)
        c21 = s.rolling(DAILY_WINDOW).sum()
        base = c21.shift(DAILY_WINDOW)       # scored window never overlaps its null
        mu = base.rolling(DAILY_BASELINE, min_periods=DAILY_MIN_OBS).mean()
        sd = base.rolling(DAILY_BASELINE, min_periods=DAILY_MIN_OBS).std()
        z = (c21 - mu) / sd.replace(0, np.nan)
        out.append(pd.DataFrame({
            "ticker": t,
            "date": cal,
            "evt8k_freq_z_d": z.values,
            "evt8k_count_21d": c21.values,
        }))
    df = pd.concat(out, ignore_index=True)
    df = df.dropna(subset=["evt8k_freq_z_d"]).sort_values(
        ["ticker", "date"]).reset_index(drop=True)

    if events is None and calendar is None:      # only cache real runs
        path = CACHE_DIR / "event_freq_8k_daily.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(
            f"Saved {len(df)} daily rows ({df['ticker'].nunique()} tickers, "
            f"{df['date'].min().date()}→{df['date'].max().date()}) to {path}"
        )
    return df


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--daily", action="store_true",
                   help="build the C15 daily variant instead of monthly C11")
    args = p.parse_args()
    df = compute_daily() if args.daily else compute_all()
    if not df.empty:
        print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
