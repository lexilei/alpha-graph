"""C32 `earnings_streak` — Loh-Warachka (2012) surprise streaks on SRW SUE.

Registered 2026-07-20 (earnings-events family member 1; construction,
adaptation, and sign pinned in reports/factor_preregistration.md before
computation):

OSAP's construction keeps only announcements whose surprise sign MATCHES
the prior announcement's and uses the surprise itself as the signal;
non-streak names exit the cross-section; announcements age out after 6
months. We have no IBES, so the surprise measure is C17's
seasonal-random-walk SUE (sue_pead.parquet) — mechanism faithful,
measurement adapted (pinned).

Per ticker in period_end order:
- classifiable iff the immediately prior defined-SUE announcement has a
  period_end gap ≤ 190 calendar days (a skipped/undefined quarter breaks
  the chain);
- signal = SUE if sign(SUE) == sign(prior SUE) and SUE != 0, else NaN
  (non-streak / unclassifiable → the NaN row BLANKS any carried value:
  the FactorSource merges with dropna=None). NOTE the 190cd adjacency is
  the WHOLE rule: when quarter-ends sit ≤190cd apart across an undefined
  quarter (fiscal transitions, ~0.4% of pairs), the defined neighbors
  still pair — an undefined quarter does not per se break the chain
  (2026-07-20 audit F2 docstring correction; code always matched the pin);
- a value row emits a NaN expiry row at disclosure_date + 182 calendar
  days ONLY if that date precedes the ticker's next announcement row of
  any kind — a later announcement always supersedes state, so expiries
  bind exactly and only for >182cd announcement silences (2026-07-20
  audit F1 fix, pinned in the ledger before computation: the original
  builder never canceled expiries and blanked 39.8% of values early).
  Same-day collisions keep the announcement row; sorts are stable so
  the later period_end wins ties deterministically (audit F3).

Availability = C17's window-capped disclosure_date ("filing" merge,
v0 t+1). Sign hypothesis: POSITIVE (streak continuation).

Build: `.venv/bin/python -m alpha_graph.signals.earnings_streak`
→ data/cache/earnings_streak.parquet (ticker, avail_date,
earnings_streak).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha_graph.config import CACHE_DIR

logger = logging.getLogger(__name__)

MAX_GAP_DAYS = 190
EXPIRY_DAYS = 182
OUT_NAME = "earnings_streak.parquet"


def build_earnings_streak(sue: pd.DataFrame) -> pd.DataFrame:
    """(ticker, avail_date, earnings_streak) rows; NaN rows are blankers."""
    df = sue.dropna(subset=["sue"]).copy()
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df = df.sort_values(["ticker", "period_end"])

    g = df.groupby("ticker")
    prev_sue = g["sue"].shift(1)
    prev_end = g["period_end"].shift(1)
    gap = (df["period_end"] - prev_end).dt.days
    adjacent = prev_sue.notna() & (gap <= MAX_GAP_DAYS)
    streak = (
        adjacent
        & (np.sign(df["sue"]) == np.sign(prev_sue))
        & (df["sue"] != 0.0)
        & (prev_sue != 0.0)
    )
    rows = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "avail_date": df["disclosure_date"],
            "earnings_streak": df["sue"].where(streak),
        }
    )
    # expiry rows: only where the +182cd date PRECEDES the ticker's next
    # announcement of any kind (audit F1 fix — a later announcement always
    # supersedes state; expiries exist for announcement silences only)
    ann = rows.sort_values(["ticker", "avail_date"], kind="mergesort")
    next_avail = ann.groupby("ticker")["avail_date"].shift(-1)
    exp_date = ann["avail_date"] + pd.Timedelta(days=EXPIRY_DAYS)
    keep = ann["earnings_streak"].notna() & (
        next_avail.isna() | (exp_date < next_avail)
    )
    expiry = ann[keep].copy()
    expiry["avail_date"] = expiry["avail_date"] + pd.Timedelta(days=EXPIRY_DAYS)
    expiry["earnings_streak"] = np.nan
    out = pd.concat([rows, expiry], ignore_index=True)
    # same-(ticker, day) collisions keep the announcement row; stable sort
    # so the later period_end (later in `rows` order) wins ties
    out["_rank"] = out["earnings_streak"].notna()
    out = (
        out.sort_values(["ticker", "avail_date", "_rank"], kind="mergesort")
        .drop_duplicates(["ticker", "avail_date"], keep="last")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sue = pd.read_parquet(CACHE_DIR / "sue_pead.parquet")
    res = build_earnings_streak(sue)
    vals = res["earnings_streak"].notna()
    out = CACHE_DIR / OUT_NAME
    res.to_parquet(out, index=False)
    logger.info(
        "earnings_streak: %d rows (%d streak values / %d blankers), "
        "%d tickers, streak share of classifiable announcements: see look row",
        len(res), int(vals.sum()), int((~vals).sum()), res["ticker"].nunique(),
    )


if __name__ == "__main__":
    main()
