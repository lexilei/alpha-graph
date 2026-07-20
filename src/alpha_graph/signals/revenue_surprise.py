"""C33 `revenue_surprise` — Jegadeesh-Livnat (2006) per the OSAP construction.

Registered 2026-07-20 (earnings-events member 2, the family's last
budgeted primary; construction and sign pinned in
reports/factor_preregistration.md before computation):

RPS_q = quarterly revenue / nearest first-filed cover-page share count
(±60 calendar days of ddate; BRK-B excluded — its only series is a
Class-A-equivalent unit, C25's pin). Quarterly revenue per (cik,
quarter) by pinned tag priority Revenues →
RevenueFromContractWithCustomerExcludingAssessedTax → SalesRevenueNet
(first-filed per (cik, ddate, qtrs); same economic quantity across the
ASC 606 transition — the mix is logged), run through
`sue_pead.quarterly_eps` for the derived-Q4 discipline.

Δ_q = RPS_q − RPS_{q−4} (the row ~365±45d earlier);
RS_q = (Δ_q − mean(prior 8 Δ)) / std(prior 8 Δ) — drift-adjusted per
OSAP ("minus the average 4-quarter change over the previous 2 years,
scaled by its std"); the prior window is the 8 immediately preceding
quarter rows, min 6 defined Δ, the current Δ excluded from both moments.

Availability = the sue cache's window-capped disclosure_date joined per
(ticker, period_end=ddate), fallback the revenue value's own
stmt_filed; "filing" merge (v0 t+1). Sign hypothesis: POSITIVE.

Build: `.venv/bin/python -m alpha_graph.signals.revenue_surprise`
→ data/cache/revenue_surprise.parquet (ticker, avail_date,
revenue_surprise).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha_graph.config import CACHE_DIR
from alpha_graph.signals.sue_pead import first_filed, quarterly_eps

logger = logging.getLogger(__name__)

TAG_PRIORITY = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
)
SHARE_MATCH_DAYS = 60
SEASON_MIN, SEASON_MAX = 320, 410
PRIOR_WINDOW = 8
PRIOR_MIN = 6
OUT_NAME = "revenue_surprise.parquet"


def merge_by_priority(facts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """First-filed revenue rows per (cik, ddate, qtrs), higher-priority tag
    wins; returns the merged frame + the tag mix over kept rows."""
    merged = None
    for tag in TAG_PRIORITY:
        f = first_filed(facts, tag, uom="USD").copy()
        f["src_tag"] = tag
        if merged is None:
            merged = f
        else:
            key = ["cik", "ddate", "qtrs"]
            existing = merged.set_index(key).index
            add = f[~f.set_index(key).index.isin(existing)]
            merged = pd.concat([merged, add], ignore_index=True)
    mix = merged["src_tag"].value_counts(normalize=True).round(3).to_dict()
    return merged, mix


def rps_join(quarters: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Revenue per share: nearest share-count `end` within ±60d of ddate."""
    q = quarters.copy()
    q["ddate"] = pd.to_datetime(q["ddate"]).astype("datetime64[ns]")
    s = shares[~shares["ticker"].str.startswith("BRK")].copy()
    s["end"] = pd.to_datetime(s["end"]).astype("datetime64[ns]")
    s = s.sort_values("end")
    q = q.sort_values("ddate")
    out = pd.merge_asof(
        q, s[["ticker", "end", "shares"]],
        left_on="ddate", right_on="end", by="ticker",
        direction="nearest", tolerance=pd.Timedelta(days=SHARE_MATCH_DAYS),
    )
    out["rps"] = out["eps"] / out["shares"]  # quarterly_eps is value-agnostic
    return out.dropna(subset=["rps"])


def revenue_surprise_from_rps(rps: pd.DataFrame) -> pd.DataFrame:
    """Drift-adjusted seasonal surprise per (ticker, quarter)."""
    rows = []
    for ticker, g in rps.groupby("ticker"):
        g = g.sort_values("ddate").reset_index(drop=True)
        dd = g["ddate"].to_numpy()
        vals = g["rps"].to_numpy()
        delta = np.full(len(g), np.nan)
        for i in range(len(g)):
            gap = (dd[i] - dd[:i]).astype("timedelta64[D]").astype(int)
            cand = np.where((gap >= SEASON_MIN) & (gap <= SEASON_MAX))[0]
            if len(cand):
                # nearest to 365d
                j = cand[np.argmin(np.abs(gap[cand] - 365))]
                delta[i] = vals[i] - vals[j]
        for i in range(len(g)):
            if np.isnan(delta[i]):
                continue
            prior = delta[max(0, i - PRIOR_WINDOW): i]
            prior = prior[~np.isnan(prior)]
            if len(prior) < PRIOR_MIN:
                continue
            sd = prior.std(ddof=1)
            if not sd > 0:
                continue
            rows.append(
                {"ticker": ticker, "ddate": g["ddate"].iloc[i],
                 "stmt_filed": g["stmt_filed"].iloc[i],
                 "revenue_surprise": (delta[i] - prior.mean()) / sd}
            )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    facts = pd.read_parquet(CACHE_DIR / "xbrl_facts.parquet")
    merged, mix = merge_by_priority(facts)
    logger.info("revenue tag mix (first-filed rows): %s", mix)
    quarters = quarterly_eps(merged)
    shares = pd.read_parquet(CACHE_DIR / "shares_outstanding_pit.parquet")
    rps = rps_join(quarters, shares)
    logger.info("quarters with RPS: %d / %d (share-match ±%dd)",
                len(rps), len(quarters), SHARE_MATCH_DAYS)
    rs = revenue_surprise_from_rps(rps)
    sue = pd.read_parquet(CACHE_DIR / "sue_pead.parquet")[
        ["ticker", "period_end", "disclosure_date"]]
    sue["period_end"] = pd.to_datetime(sue["period_end"])
    rs = rs.merge(sue, left_on=["ticker", "ddate"],
                  right_on=["ticker", "period_end"], how="left")
    n_sue = rs["disclosure_date"].notna().sum()
    rs["avail_date"] = rs["disclosure_date"].fillna(rs["stmt_filed"])
    rs["avail_date"] = pd.to_datetime(rs["avail_date"])
    out = rs[["ticker", "avail_date", "revenue_surprise"]].dropna()
    out = out.sort_values(["ticker", "avail_date"]).reset_index(drop=True)
    path = CACHE_DIR / OUT_NAME
    out.to_parquet(path, index=False)
    logger.info(
        "revenue_surprise: %d rows / %d tickers, %s -> %s; "
        "disclosure via sue cache %.1f%%, stmt_filed fallback %.1f%%",
        len(out), out["ticker"].nunique(),
        out["avail_date"].min().date(), out["avail_date"].max().date(),
        100 * n_sue / len(rs), 100 * (1 - n_sue / len(rs)),
    )


if __name__ == "__main__":
    main()
