"""C30 `volume_trend` — OSAP-faithful Haugen-Baker (1996) volume trend.

Registered 2026-07-17 (liquidity-volume family member 1; construction and
sign pinned in reports/factor_preregistration.md before computation):

At each month-end, OLS slope of monthly SHARE volume on a linear time
index over the trailing 60 calendar months (min 30 defined), scaled by
the window's mean volume. Sign hypothesis: NEGATIVE (high volume trend →
low forward returns; OSAP Sign=−1).

Pinned details:
- Monthly volume = sum of daily share volume per (ticker, calendar month).
  market_data volume is back-adjusted (split-continuous); slope/mean is
  invariant to uniform per-ticker scaling, so the basis is immaterial.
- Partial ticker-months excluded: a month where the ticker has fewer than
  15 trading days of data while the panel calendar has more (panel edges,
  IPOs, re-adds) would fake a volume dip at the window edge.
- The 60-month window is CALENDAR months (missing months stay missing,
  they don't compress the time axis); slope uses each defined month's
  position in the window as x.
- Rows are emitted only at defined ticker-months, stamped with the
  ticker's last trading day of that month (computable at that close);
  merge kind "grid" → v0 t+1 on the daily grid.

Known limitations (2026-07-20 audit; behavior deliberately unchanged —
the C30 look is recorded on this exact construction):
- volume==0 days count as trading days (a data hole reads as a volume
  collapse; HWM 2016-11/12 is the one real case);
- the panel-edge partial month at the very end of the panel is KEPT per
  the pinned rule (n_days == panel_days there), which level-shifts the
  final cross-section; it never feeds a forward return.
- Post-2020 evaluation context: ~1/3 of the H2 IC is COVID window-transit
  mechanics and the rest correlates −0.42 with trailing-60m price return
  (uncontrolled by the judge) — see the ledger's amended rescue clause.

Build: `.venv/bin/python -m alpha_graph.signals.volume_trend`
→ data/cache/volume_trend.parquet (ticker, date, volume_trend).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha_graph.config import CACHE_DIR

logger = logging.getLogger(__name__)

WINDOW_MONTHS = 60
MIN_MONTHS = 30
MIN_DAYS_FULL_MONTH = 15
OUT_NAME = "volume_trend.parquet"


def _monthly_volume(market: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, month): summed share volume, day count, month-end date;
    partial ticker-months (vs the panel calendar) dropped."""
    df = market[["ticker", "date", "volume"]].dropna(subset=["volume"]).copy()
    df["month"] = df["date"].dt.to_period("M")
    panel_days = df.groupby("month")["date"].nunique().rename("panel_days")
    m = (
        df.groupby(["ticker", "month"])
        .agg(vol=("volume", "sum"), n_days=("date", "nunique"),
             date=("date", "max"))
        .reset_index()
        .join(panel_days, on="month")
    )
    partial = (m["n_days"] < MIN_DAYS_FULL_MONTH) & (m["n_days"] < m["panel_days"])
    return m[~partial][["ticker", "month", "vol", "date"]]


def _trend(vols: np.ndarray) -> float:
    """OLS slope of vol on its month position within the window, scaled by
    the window mean; NaN entries = missing calendar months."""
    x = np.arange(len(vols), dtype=float)
    ok = ~np.isnan(vols)
    if ok.sum() < MIN_MONTHS:
        return np.nan
    x, y = x[ok], vols[ok]
    mean = y.mean()
    if not mean > 0:
        return np.nan
    xc = x - x.mean()
    denom = (xc**2).sum()
    if denom == 0:
        return np.nan
    slope = (xc * (y - mean)).sum() / denom
    return slope / mean


def build_volume_trend(market: pd.DataFrame) -> pd.DataFrame:
    """(ticker, date, volume_trend) at defined ticker-month-ends."""
    monthly = _monthly_volume(market)
    out = []
    for ticker, g in monthly.groupby("ticker"):
        g = g.set_index("month").sort_index()
        full = g.reindex(pd.period_range(g.index.min(), g.index.max(), freq="M"))
        vols = full["vol"].to_numpy(dtype=float)
        for i in range(len(full)):
            if np.isnan(vols[i]):
                continue  # rows only at defined ticker-months
            w = vols[max(0, i - WINDOW_MONTHS + 1): i + 1]
            vt = _trend(w)
            if np.isnan(vt):
                continue
            out.append(
                {"ticker": ticker, "date": full["date"].iloc[i],
                 "volume_trend": vt}
            )
    return pd.DataFrame(out, columns=["ticker", "date", "volume_trend"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    market = pd.read_parquet(CACHE_DIR / "market_data.parquet")
    market["date"] = pd.to_datetime(market["date"])
    res = build_volume_trend(market)
    out = CACHE_DIR / OUT_NAME
    res.to_parquet(out, index=False)
    logger.info(
        "volume_trend: %d rows / %d tickers, %s -> %s",
        len(res), res["ticker"].nunique(),
        res["date"].min().date(), res["date"].max().date(),
    )


if __name__ == "__main__":
    main()
