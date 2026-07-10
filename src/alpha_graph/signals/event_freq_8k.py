"""8-K event-frequency factors (C11 family).

The predictive content of 8-K filings in this universe is in the *abnormal
frequency* of filing activity, not in item content or tone: a spike in a firm's
8-K count relative to its own trailing baseline predicts lower forward returns
(the market under-reacts to clustered disclosure). Item-type and sentiment
variants were tested and rejected — see FACTORS.md / the pre-registration log.

Factor `evt8k_freq_z`: z-score of the current calendar month's 8-K count vs the
firm's trailing 24-month count distribution. PIT (uses only months ≤ t). NaN
for tickers with no 8-K corpus (missing data, not "zero events").

Usage:
    python -m alpha_graph.signals.event_freq_8k
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR

BASELINE_MONTHS = 24     # trailing window for the abnormal-frequency z-score
MIN_MONTHS = 6           # need this many trailing months before scoring


def _filing_months() -> pd.DataFrame:
    """One row per 8-K filing: ticker, filing month (Period[M])."""
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
    df = df.dropna(subset=["fdate"])
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


def main():
    df = compute_all()
    if not df.empty:
        print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
