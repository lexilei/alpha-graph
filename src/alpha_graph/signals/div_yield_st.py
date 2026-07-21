"""C36 `div_yield_st` — Litzenberger-Ramaswamy predicted dividend yield.

Registered 2026-07-21 (div-seasonality member 1; construction, sign,
and FAST persistence class pinned in reports/factor_preregistration.md
before computation; the family's cost prerequisite ran first):

Per (ticker, month m), universe = paid ≥1 dividend in trailing 12
months. Payment frequency inferred POINT-IN-TIME from the trailing 36
months of ex-dates as of m (median gap: <135d → Q, <270d → S, else A;
<2 gaps → unclassifiable → NaN). Expected dividend next month
Ediv1 = the amount at the ex-date 2 months ago (Q) / 5 (S) / 11 (A),
else 0.0. Signal = Ediv1 / month-end close (dividends and closes share
the yfinance back-adjusted split basis; zeros kept, non-payers NaN).

Only PAST ex-dates enter — PIT-safe by construction. Month-end stamped
at the ticker's last trading day, "grid" merge (v0 t+1: the signal
predicts month m+1's ex-div, consumed against fwd_return_21d).

Build: `.venv/bin/python -m alpha_graph.signals.div_yield_st`
→ data/cache/div_yield_st.parquet (ticker, date, div_yield_st).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha_graph.config import CACHE_DIR

logger = logging.getLogger(__name__)

FREQ_WINDOW_MONTHS = 36
Q_MAX_GAP, S_MAX_GAP = 135, 270
LAG_BY_FREQ = {"Q": 2, "S": 5, "A": 11}
OUT_NAME = "div_yield_st.parquet"


def _infer_freq(ex_dates: np.ndarray, asof: pd.Timestamp) -> str | None:
    """Median gap of ex-dates in the trailing 36 months as of `asof`."""
    lo = asof - pd.DateOffset(months=FREQ_WINDOW_MONTHS)
    win = ex_dates[(ex_dates > lo) & (ex_dates <= asof)]
    if len(win) < 3:
        return None
    gaps = np.diff(win).astype("timedelta64[D]").astype(int)
    med = float(np.median(gaps))
    if med < Q_MAX_GAP:
        return "Q"
    if med < S_MAX_GAP:
        return "S"
    return "A"


def build_div_yield_st(div: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    div = div.copy()
    div["ex_month"] = div["ex_date"].dt.to_period("M")
    # one amount per (ticker, ex_month): sum (specials add cash, rare)
    dm = div.groupby(["ticker", "ex_month"])["amount"].sum()
    ex_by_ticker = {t: g["ex_date"].sort_values().to_numpy()
                    for t, g in div.groupby("ticker")}

    px = market[["ticker", "date", "close"]].copy()
    px["month"] = px["date"].dt.to_period("M")
    month_end = (
        px.sort_values("date").groupby(["ticker", "month"], as_index=False)
        .tail(1)
    )

    rows = []
    for t, g in month_end.groupby("ticker"):
        ex_dates = ex_by_ticker.get(t)
        if ex_dates is None:
            continue
        for _, r in g.iterrows():
            m = r["month"]
            paid_12m = any((t, m - k) in dm.index for k in range(1, 13))
            if not paid_12m:
                continue
            freq = _infer_freq(ex_dates, r["date"])
            if freq is None:
                continue
            lag = LAG_BY_FREQ[freq]
            ediv1 = float(dm.get((t, m - lag), 0.0))
            if not r["close"] > 0:
                continue
            rows.append(
                {"ticker": t, "date": r["date"],
                 "div_yield_st": ediv1 / float(r["close"])}
            )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    div = pd.read_parquet(CACHE_DIR / "dividends.parquet")
    market = pd.read_parquet(CACHE_DIR / "market_data.parquet")
    market["date"] = pd.to_datetime(market["date"])
    res = build_div_yield_st(div, market)
    path = CACHE_DIR / OUT_NAME
    res.to_parquet(path, index=False)
    nz = res["div_yield_st"] > 0
    logger.info(
        "div_yield_st: %d rows / %d tickers, %s -> %s; nonzero %.1f%%, "
        "median nonzero yield %.4f",
        len(res), res["ticker"].nunique(),
        res["date"].min().date(), res["date"].max().date(),
        100 * nz.mean(), res.loc[nz, "div_yield_st"].median(),
    )


if __name__ == "__main__":
    main()
