"""C34 `short_interest_xs` — FINRA short interest ratio (Dechow et al 2001).

Registered 2026-07-20 (ownership-flow family member 1; construction and
sign pinned in reports/factor_preregistration.md before computation):

SI ratio = FINRA consolidated bi-monthly short position ÷
`shares_pit.shares_asof(ticker, avail_date)` — the latest share count
FILED on or before the availability date (the C33-audit standing
clause: no nearest-end matching, the denominator is never a count the
market hadn't seen). avail_date = settlement + 10 business days,
stamped by the fetcher (conservative vs FINRA's ~7-9bd dissemination).

Symbol map: FINRA compresses class shares — {BRKB→BRK-B, BFB→BF-B};
BRK-B is then excluded anyway (its only share series is Class-A-
equivalent cap units — the B7/C25 pin), so it ships NaN. Rows with no
count filed yet are dropped. Sign hypothesis: NEGATIVE (high SI → low
forward returns).

Build: `.venv/bin/python -m alpha_graph.signals.short_interest_xs`
→ data/cache/short_interest_xs.parquet (ticker, avail_date,
short_interest_xs).
"""
from __future__ import annotations

import logging

import pandas as pd

from alpha_graph.config import CACHE_DIR
from alpha_graph.data.shares_pit import shares_asof

logger = logging.getLogger(__name__)

SYMBOL_MAP = {"BRKB": "BRK-B", "BFB": "BF-B"}
EXCLUDE = {"BRK-B"}  # cap-unit share series, wrong denominator units
OUT_NAME = "short_interest_xs.parquet"


def build_short_interest_xs(
    si: pd.DataFrame, panel_tickers: set[str]
) -> pd.DataFrame:
    """(ticker, avail_date, short_interest_xs) for panel tickers."""
    df = si.copy()
    df["ticker"] = df["symbol"].replace(SYMBOL_MAP)
    df = df[df["ticker"].isin(panel_tickers - EXCLUDE)]
    df["avail_date"] = pd.to_datetime(df["avail_date"])
    out = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("avail_date")
        try:
            sh = shares_asof(ticker, g["avail_date"])
        except Exception:  # noqa: BLE001 - no share facts for this ticker
            continue
        ratio = g["short_qty"].to_numpy() / sh.to_numpy()
        out.append(
            pd.DataFrame(
                {"ticker": ticker, "avail_date": g["avail_date"].to_numpy(),
                 "short_interest_xs": ratio}
            )
        )
    if not out:
        return pd.DataFrame(columns=["ticker", "avail_date", "short_interest_xs"])
    res = pd.concat(out, ignore_index=True).dropna()
    return res.sort_values(["ticker", "avail_date"]).reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    si = pd.read_parquet(CACHE_DIR / "short_interest.parquet")
    market = pd.read_parquet(CACHE_DIR / "market_data.parquet")
    res = build_short_interest_xs(si, set(market["ticker"].unique()))
    path = CACHE_DIR / OUT_NAME
    res.to_parquet(path, index=False)
    logger.info(
        "short_interest_xs: %d rows / %d tickers, %s -> %s, "
        "median ratio %.4f, p99 %.4f",
        len(res), res["ticker"].nunique(),
        res["avail_date"].min().date(), res["avail_date"].max().date(),
        res["short_interest_xs"].median(), res["short_interest_xs"].quantile(0.99),
    )


if __name__ == "__main__":
    main()
