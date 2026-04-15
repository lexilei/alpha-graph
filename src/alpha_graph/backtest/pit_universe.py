"""Point-in-time S&P 500 membership filter for Method E.

The baseline universe in this repo is 499 current S&P 500 constituents. That
introduces two kinds of survivorship bias:

  (a) FORWARD bias: tickers that are in the 2026 index but were not in the
      index during earlier years of the backtest (e.g. NVDA's index
      membership rose over time; a 2012 pick includes firms that later
      joined the index). This IS fixable from existing market data — we
      just need PIT membership and then filter the candidate pool per month.

  (b) SURVIVOR bias: tickers that were removed from the index before 2026
      and dropped by the downloader are completely missing. Fixing this
      requires refetching filings + market data for delisted tickers, which
      is a separate data-engineering task.

This script addresses (a) only. It downloads the historical S&P 500 membership
table from https://github.com/fja05680/sp500, merges it with the existing ML
combiner predictions, and reruns Method E restricted to tickers that were in
the index at each month.

If Method E's Sharpe survives the PIT-universe filter, we've ruled out
forward-bias (a) as the explanation. If it degrades significantly, we've
quantified the forward-bias component of the inflation.

Usage:
    python -m alpha_graph.backtest.pit_universe
"""

from __future__ import annotations

import io
import urllib.request

import pandas as pd
from loguru import logger

from alpha_graph.backtest.extensions import _load_data, _metrics
from alpha_graph.config import CACHE_DIR, set_global_seeds

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv"
)


def fetch_membership() -> pd.DataFrame:
    cache = CACHE_DIR / "sp500_pit_membership.csv"
    if cache.exists():
        logger.info(f"Using cached PIT membership at {cache}")
        return pd.read_csv(cache, parse_dates=["date"])

    logger.info("Fetching historical S&P 500 membership from GitHub")
    req = urllib.request.Request(MEMBERSHIP_URL, headers={"User-Agent": "research"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read().decode()
    df = pd.read_csv(io.StringIO(data), parse_dates=["date"])
    df.to_csv(cache, index=False)
    logger.info(f"Cached PIT membership to {cache} ({len(df)} snapshots)")
    return df


def membership_for_month(membership: pd.DataFrame, month: pd.Period) -> set[str]:
    """Return the set of tickers in the index at the end of the given month."""
    month_end = month.to_timestamp(how="end")
    at_or_before = membership[membership["date"] <= month_end]
    if at_or_before.empty:
        return set()
    latest = at_or_before.iloc[-1]["tickers"]
    return set(latest.split(","))


def filter_preds_pit(preds: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Drop any ticker-month row where the ticker was not in the S&P 500 at
    the end of the prior month (we use prior-month membership so the universe
    decision is strictly PIT — no lookahead).
    """
    membership = membership.sort_values("date").reset_index(drop=True)
    preds = preds.copy()

    if not isinstance(preds["month"].dtype, pd.PeriodDtype):
        preds["month"] = pd.to_datetime(preds["date"]).dt.to_period("M")

    months = sorted(preds["month"].unique())
    kept = []
    dropped_rows = 0
    for month in months:
        prior_end = (month - 1).to_timestamp(how="end")
        at_or_before = membership[membership["date"] <= prior_end]
        if at_or_before.empty:
            continue
        members = set(at_or_before.iloc[-1]["tickers"].split(","))
        mask = preds["month"] == month
        block = preds[mask]
        keep_mask = block["ticker"].isin(members)
        dropped_rows += int((~keep_mask).sum())
        kept.append(block[keep_mask])
    pit_preds = pd.concat(kept, ignore_index=True) if kept else preds.iloc[0:0]
    logger.info(
        f"PIT filter: kept {len(pit_preds):,} rows, "
        f"dropped {dropped_rows:,} rows ({dropped_rows / len(preds) * 100:.1f}% of original)"
    )
    return pit_preds


def method_e_backtest(preds: pd.DataFrame, cost_per_month: float = 0.001) -> dict:
    months = sorted(preds["month"].unique())
    rows = []
    for month in months:
        m = preds[preds["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue
        sorted_m = m.sort_values("signal", ascending=False)
        long_r = float(sorted_m.head(10)["actual_return"].mean())
        rows.append({
            "month": month.to_timestamp(),
            "net": long_r - cost_per_month,
        })
    df = pd.DataFrame(rows)
    return _metrics(df)


def main():
    set_global_seeds()
    preds, _market, _regimes = _load_data()
    membership = fetch_membership()

    # Baseline (all 499 current constituents)
    base = method_e_backtest(preds)

    # PIT-filtered
    pit_preds = filter_preds_pit(preds, membership)
    pit = method_e_backtest(pit_preds)

    # Per-month universe size — diagnostic
    preds["month"] = pd.to_datetime(preds["date"]).dt.to_period("M")
    before_sizes = preds.groupby("month")["ticker"].nunique()
    after_sizes = pit_preds.groupby("month")["ticker"].nunique() if not pit_preds.empty else pd.Series(dtype=int)

    print()
    print("=" * 78)
    print("  POINT-IN-TIME UNIVERSE: METHOD E SURVIVORSHIP (FORWARD-BIAS) CHECK")
    print("=" * 78)
    print(
        f"  {'Variant':<40s}  {'Sharpe':>7s}  {'Ann':>7s}  {'Cum':>9s}  {'MaxDD':>7s}"
    )
    print("-" * 78)
    print(
        f"  {'Method E — current 499 universe':<40s}  "
        f"{base['sharpe']:>+6.2f}  {base['ann']:>+6.1%}  {base['cum']:>+8.1%}  {base['maxdd']:>+6.1%}"
    )
    print(
        f"  {'Method E — PIT S&P 500 universe':<40s}  "
        f"{pit['sharpe']:>+6.2f}  {pit['ann']:>+6.1%}  {pit['cum']:>+8.1%}  {pit['maxdd']:>+6.1%}"
    )
    print("-" * 78)
    print(f"  Sharpe delta:  {pit['sharpe'] - base['sharpe']:+.2f}")
    print(f"  Ann return delta: {(pit['ann'] - base['ann'])*100:+.2f} pp")
    print()
    print("  Universe size per month (min / median / max):")
    print(f"    Current constituents: {before_sizes.min()} / {int(before_sizes.median())} / {before_sizes.max()}")
    if not after_sizes.empty:
        print(f"    PIT members:          {after_sizes.min()} / {int(after_sizes.median())} / {after_sizes.max()}")
    print("=" * 78)
    print()
    print("  Caveat: this only fixes FORWARD bias (tickers that entered the index")
    print("  after the month being backtested). The SURVIVOR bias — tickers that")
    print("  were removed from the index before 2026 and are entirely missing from")
    print("  the 499-ticker download — is NOT fixed here and remains an open caveat.")
    print("=" * 78)

    # Save
    pd.DataFrame([
        {"variant": "current_universe", **base},
        {"variant": "pit_universe", **pit},
    ]).to_parquet(CACHE_DIR / "method_e_pit_comparison.parquet", index=False)
    logger.info("Saved PIT comparison")


if __name__ == "__main__":
    main()
