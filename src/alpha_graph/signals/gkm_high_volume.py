"""C31 `gkm_high_volume` — GKM (2001) high-volume return premium, event form.

Registered 2026-07-18 (liquidity-volume member 2; construction and sign
pinned in reports/factor_preregistration.md before computation):

Daily signal in {+1, 0, -1}: +1 if the day's share volume exceeds the
90th percentile of that ticker's PRIOR 49 trading days' volumes, -1 if
below the 10th percentile, else 0. The formation day is excluded from
its own reference window; a full 49-day window of defined volume is
required (else NaN → row not emitted). Stamped at the formation day,
merge kind "grid" → v0 t+1 = GKM's next-day portfolio formation; the
platform's 21d forward target sits inside GKM's 20-100d premium window.

Sign hypothesis: POSITIVE (visibility — extreme volume attracts
attention → buying pressure → drift; the low-volume leg mirrors).

Percentiles are pandas rolling quantiles (linear interpolation).

Build: `.venv/bin/python -m alpha_graph.signals.gkm_high_volume`
→ data/cache/gkm_high_volume.parquet (ticker, date, gkm_high_volume).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha_graph.config import CACHE_DIR

logger = logging.getLogger(__name__)

WINDOW = 49
HI_Q = 0.90
LO_Q = 0.10
OUT_NAME = "gkm_high_volume.parquet"


def build_gkm(market: pd.DataFrame) -> pd.DataFrame:
    """(ticker, date, gkm_high_volume) at days with a full prior window."""
    df = (
        market[["ticker", "date", "volume"]]
        .dropna(subset=["volume"])
        .sort_values(["ticker", "date"])
        .copy()
    )
    g = df.groupby("ticker")["volume"]
    prior = g.shift(1)  # formation day excluded from its own reference
    roll = prior.groupby(df["ticker"]).rolling(WINDOW, min_periods=WINDOW)
    df["q_hi"] = roll.quantile(HI_Q).reset_index(level=0, drop=True)
    df["q_lo"] = roll.quantile(LO_Q).reset_index(level=0, drop=True)
    df = df.dropna(subset=["q_hi", "q_lo"])
    sig = np.zeros(len(df))
    sig[df["volume"].to_numpy() > df["q_hi"].to_numpy()] = 1.0
    sig[df["volume"].to_numpy() < df["q_lo"].to_numpy()] = -1.0
    df["gkm_high_volume"] = sig
    return df[["ticker", "date", "gkm_high_volume"]].reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    market = pd.read_parquet(CACHE_DIR / "market_data.parquet")
    market["date"] = pd.to_datetime(market["date"])
    res = build_gkm(market)
    share = res["gkm_high_volume"].value_counts(normalize=True).round(4)
    out = CACHE_DIR / OUT_NAME
    res.to_parquet(out, index=False)
    logger.info(
        "gkm_high_volume: %d rows / %d tickers, %s -> %s; shares %s",
        len(res), res["ticker"].nunique(),
        res["date"].min().date(), res["date"].max().date(), share.to_dict(),
    )


if __name__ == "__main__":
    main()
