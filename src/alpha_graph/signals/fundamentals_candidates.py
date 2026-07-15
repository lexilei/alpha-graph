"""C25/C26/C27: three fundamentals-family candidates on data already in hand.

Registered 2026-07-15 (reports/factor_preregistration.md row "C25/C26/C27
registration", commit `28b84c6`; FACTORS.md C25/C26/C27). Family=XS-IC,
roles=sel, ONE primary v0 look each, bar sn incr |t| >= 1.5 -> candidate else
rejected. Each signal writes its own per-signal cache and merges as a
filing-date FactorSource (availability = the stamped date, v0 t+1 lag applied
at the panel merge, like every other filing-dated factor).

- **C25 `net_issuance_12m`** (Pontiff-Woodgate; issuance -> low returns,
  buybacks -> high). 12-month log change in shares outstanding from
  `shares_outstanding_pit.parquet`. Per ticker, the FIRST-FILED share count
  per cover-page `end` date (classes summed same-(end,filed) first, then the
  earliest `filed` per end), ordered by `end`; signal at obs i = log(shares_i /
  shares_j) where j is the observation ~365 +/-45 days earlier (nearest to
  365). Availability = the LATER observation's filed date. BRK-B excluded (its
  only series is a Class-A-equivalent weighted average — wrong units; see
  data/shares_pit.CAP_UNIT_MISMATCH_TICKERS). SPLIT RULE (pinned, simpler
  robust form): a raw count is NOT split-adjusted, so a stock split shows as a
  >1.9x (or a reverse split / data artifact as a <0.55x) step between
  consecutive observations; flag those steps and NaN any 12m window that
  CONTAINS one (a step at position k NaNs every window (j, i] with j < k <= i).
  The builder logs which steps were flagged and how many windows got NaN'd.

- **C26 `asset_growth_yoy`** (Cooper-Gulen-Schill; high growth -> low returns).
  YoY log change in as-filed total `Assets` (qtrs=0 instants) from
  `xbrl_facts.parquet`, first-filed per (cik, ddate) via `sue_pead.first_filed`
  (imported, not reimplemented). Signal at ddate i = log(assets_i / assets_j),
  j the ddate ~365 +/-45 days earlier. Availability = the later accession's
  filed date.

- **C27 `ann_ret_2d`** (price-based PEAD; announcement-window return
  continues). Around each Item-2.02 8-K acceptance day, the announcement day a
  = the acceptance calendar date if accepted before 16:00 ET else the next
  trading day (both rolled onto the market calendar). Signal =
  close(a-1) -> close(a+1) simple return from the panel prices; skipped if
  either close is missing. Availability = a+1. Same-day multi-8-K dedup: keep
  the FIRST 2.02 of the day per (ticker, filing_date) — earliest acceptance_ts.

Usage:
    python -m alpha_graph.signals.fundamentals_candidates          # all three
    python -m alpha_graph.signals.fundamentals_candidates --only c25
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR
from alpha_graph.data.shares_pit import CAP_UNIT_MISMATCH_TICKERS
from alpha_graph.signals.sue_pead import first_filed

SEASONAL_DAYS = 365          # target gap to the observation ~1 year earlier
SEASONAL_TOL_DAYS = 45       # match tolerance around SEASONAL_DAYS (like SUE)
SPLIT_UP = 1.9               # consecutive count ratio above this = split-like
SPLIT_DOWN = 0.55            # ... or below this (reverse split / artifact)
ACCEPT_CUTOFF_HOUR = 16      # ET; accepted at/after this -> next-trading-day a
ITEM_202 = "2.02"

C25_NAME = "net_issuance_12m.parquet"
C26_NAME = "asset_growth_yoy.parquet"
C27_NAME = "ann_ret_2d.parquet"


# --------------------------------------------------------------------------- #
# Shared: the ~365 +/-45 day "one year earlier" match (SUE's seasonal rule)
# --------------------------------------------------------------------------- #

def _yoy_match(dates: np.ndarray) -> np.ndarray:
    """For each i, the index j (< i) of the observation ~SEASONAL_DAYS earlier
    (within +/-SEASONAL_TOL_DAYS, nearest to SEASONAL_DAYS); -1 if none.

    `dates` is a sorted-ascending datetime64[ns] array. Mirrors
    sue_pead.add_sue's searchsorted match so the 12m/YoY windows are the same
    "same period one year earlier" rule the earnings factor uses."""
    dates = dates.astype("datetime64[ns]")
    lo_gap = np.timedelta64(SEASONAL_DAYS + SEASONAL_TOL_DAYS, "D")
    hi_gap = np.timedelta64(SEASONAL_DAYS - SEASONAL_TOL_DAYS, "D")
    target = np.timedelta64(SEASONAL_DAYS, "D")
    lo = np.searchsorted(dates, dates - lo_gap, side="left")
    hi = np.searchsorted(dates, dates - hi_gap, side="right")
    out = np.full(len(dates), -1)
    for i in range(len(dates)):
        if hi[i] <= lo[i]:
            continue                                     # no match in tolerance
        cand = np.arange(lo[i], hi[i])
        out[i] = cand[np.argmin(np.abs((dates[i] - dates[cand]) - target))]
    return out


# --------------------------------------------------------------------------- #
# C25 net_issuance_12m
# --------------------------------------------------------------------------- #

def compute_net_issuance(shares: pd.DataFrame | None = None,
                         write: bool = False) -> pd.DataFrame:
    """Build the C25 cache: ticker, avail_date, end, net_issuance_12m,
    split_nan. Injectable `shares` for tests (defaults to the real cache)."""
    if shares is None:
        shares = pd.read_parquet(CACHE_DIR / "shares_outstanding_pit.parquet")

    excl = set(CAP_UNIT_MISMATCH_TICKERS)
    df = shares[~shares["ticker"].isin(excl)].copy()
    df = df.dropna(subset=["end", "filed", "shares"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    # classes split across rows for the same (end, filed): sum them (the
    # data/shares_pit collapse), then first-filed per (ticker, end).
    g = df.groupby(["ticker", "end", "filed"], as_index=False)["shares"].sum()
    ff = g.sort_values(["filed", "end"], kind="mergesort").drop_duplicates(
        ["ticker", "end"], keep="first")

    parts, split_events, n_windows_nan = [], [], 0
    for tk, gt in ff.groupby("ticker", sort=True):
        gt = gt.sort_values("end").reset_index(drop=True)
        ends = gt["end"].to_numpy().astype("datetime64[ns]")
        shr = gt["shares"].to_numpy(dtype=float)
        filed = gt["filed"].to_numpy().astype("datetime64[ns]")

        ratio = np.full(len(gt), np.nan)
        if len(gt) > 1:
            ratio[1:] = shr[1:] / shr[:-1]
        with np.errstate(invalid="ignore"):
            split_step = (ratio > SPLIT_UP) | (ratio < SPLIT_DOWN)
        cum = np.cumsum(split_step.astype(int))          # steps at positions <= k
        for k in np.flatnonzero(split_step):
            split_events.append((tk, pd.Timestamp(ends[k]), float(ratio[k])))

        match = _yoy_match(ends)
        sig = np.full(len(gt), np.nan)
        split_nan = np.zeros(len(gt), dtype=bool)
        for i in range(len(gt)):
            j = match[i]
            if j < 0 or shr[i] <= 0 or shr[j] <= 0:
                continue
            if cum[i] - cum[j] > 0:                       # a split step in (j, i]
                split_nan[i] = True
                n_windows_nan += 1
                continue
            sig[i] = np.log(shr[i] / shr[j])
        parts.append(pd.DataFrame({
            "ticker": tk, "avail_date": filed, "end": ends,
            "net_issuance_12m": sig, "split_nan": split_nan,
        }))

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["ticker", "avail_date", "end", "net_issuance_12m", "split_nan"])
    out = out.sort_values(["ticker", "avail_date", "end"],
                          kind="mergesort").reset_index(drop=True)
    defined = int(out["net_issuance_12m"].notna().sum())
    logger.info(
        f"C25 net_issuance_12m: {len(out)} obs, {out['ticker'].nunique()} "
        f"tickers, {defined} defined ({100 * defined / max(len(out), 1):.1f}%); "
        f"split rule (>{SPLIT_UP}x or <{SPLIT_DOWN}x consecutive step): "
        f"{len(split_events)} steps flagged, {n_windows_nan} 12m windows NaN'd")
    for tk, end, r in split_events:
        logger.debug(f"  C25 split step: {tk} {end.date()} ratio {r:.3f}")
    if write:
        out.to_parquet(CACHE_DIR / C25_NAME, index=False)
        logger.info(f"Saved {len(out)} rows to {CACHE_DIR / C25_NAME}")
    return out


# --------------------------------------------------------------------------- #
# C26 asset_growth_yoy
# --------------------------------------------------------------------------- #

def compute_asset_growth(facts: pd.DataFrame | None = None,
                         write: bool = False) -> pd.DataFrame:
    """Build the C26 cache: ticker, avail_date, ddate, asset_growth_yoy.
    Injectable `facts` for tests (defaults to xbrl_facts.parquet)."""
    if facts is None:
        facts = pd.read_parquet(CACHE_DIR / "xbrl_facts.parquet")

    ff = first_filed(facts, "Assets", "USD")             # one row / (cik,ddate,qtrs)
    a = ff[ff["qtrs"] == 0].copy()
    a["ddate"] = pd.to_datetime(a["ddate"])
    a["filed"] = pd.to_datetime(a["filed"])

    parts = []
    for _, g in a.groupby("cik", sort=False):
        g = g.sort_values("ddate").reset_index(drop=True)
        dd = g["ddate"].to_numpy().astype("datetime64[ns]")
        val = g["value"].to_numpy(dtype=float)
        filed = g["filed"].to_numpy().astype("datetime64[ns]")
        match = _yoy_match(dd)
        sig = np.full(len(g), np.nan)
        for i in range(len(g)):
            j = match[i]
            if j < 0 or val[i] <= 0 or val[j] <= 0:
                continue
            sig[i] = np.log(val[i] / val[j])
        parts.append(pd.DataFrame({
            "ticker": g["ticker"].to_numpy(), "avail_date": filed,
            "ddate": dd, "asset_growth_yoy": sig,
        }))

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["ticker", "avail_date", "ddate", "asset_growth_yoy"])
    out = out.sort_values(["ticker", "avail_date", "ddate"],
                          kind="mergesort").reset_index(drop=True)
    defined = int(out["asset_growth_yoy"].notna().sum())
    logger.info(
        f"C26 asset_growth_yoy: {len(out)} obs, {out['ticker'].nunique()} "
        f"tickers, {defined} defined ({100 * defined / max(len(out), 1):.1f}%)")
    if write:
        out.to_parquet(CACHE_DIR / C26_NAME, index=False)
        logger.info(f"Saved {len(out)} rows to {CACHE_DIR / C26_NAME}")
    return out


# --------------------------------------------------------------------------- #
# C27 ann_ret_2d
# --------------------------------------------------------------------------- #

def _has_202(items: str) -> bool:
    return ITEM_202 in [x.strip() for x in str(items).split(",")]


def compute_ann_ret_2d(items: pd.DataFrame | None = None,
                       acceptance: pd.DataFrame | None = None,
                       market: pd.DataFrame | None = None,
                       write: bool = False) -> pd.DataFrame:
    """Build the C27 cache: ticker, avail_date, ann_day, ann_ret_2d. All three
    inputs injectable for tests (defaults read the real caches)."""
    if items is None:
        items = pd.read_parquet(CACHE_DIR / "filing_items.parquet")
    if acceptance is None:
        acceptance = pd.read_parquet(CACHE_DIR / "filing_acceptance.parquet")
    if market is None:
        market = pd.read_parquet(CACHE_DIR / "market_data.parquet",
                                 columns=["ticker", "date", "close"])

    ev = items[items["items"].map(_has_202)].copy()
    ev["filing_date"] = pd.to_datetime(ev["filing_date"])
    ev = ev.merge(
        acceptance[["accession", "acceptance_ts"]].drop_duplicates("accession"),
        on="accession", how="left")
    ev["acceptance_ts"] = pd.to_datetime(ev["acceptance_ts"])
    n_missing = int(ev["acceptance_ts"].isna().sum())
    if n_missing:
        logger.warning(f"{n_missing} Item-2.02 8-Ks lack an acceptance "
                       f"timestamp — treated as before-16:00 (no shift)")
    # same-day multi-8-K dedup: FIRST 2.02 of the day per (ticker, filing_date)
    ev = ev.sort_values(["ticker", "filing_date", "acceptance_ts"],
                        kind="mergesort", na_position="last")
    ev = ev.drop_duplicates(["ticker", "filing_date"], keep="first")

    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])
    cal = np.sort(market["date"].unique()).astype("datetime64[ns]")

    base = ev["acceptance_ts"].dt.normalize().to_numpy().astype("datetime64[ns]")
    # NaT acceptance falls back to the filing_date, treated as before 16:00
    fd = ev["filing_date"].to_numpy().astype("datetime64[ns]")
    base = np.where(np.isnat(base), fd, base)
    hour = ev["acceptance_ts"].dt.hour.to_numpy()
    after = np.where(np.isnan(hour), False, hour >= ACCEPT_CUTOFF_HOUR)
    pos = np.where(after, np.searchsorted(cal, base, side="right"),
                   np.searchsorted(cal, base, side="left"))
    valid = (pos - 1 >= 0) & (pos + 1 < len(cal))

    d = ev.loc[valid, ["ticker"]].copy()
    pv = pos[valid]
    d["am1"] = cal[pv - 1]
    d["ann_day"] = cal[pv]
    d["ap1"] = cal[pv + 1]

    mc = market[["ticker", "date", "close"]]
    d = d.merge(mc.rename(columns={"date": "am1", "close": "c_m1"}),
                on=["ticker", "am1"], how="left")
    d = d.merge(mc.rename(columns={"date": "ap1", "close": "c_p1"}),
                on=["ticker", "ap1"], how="left")
    d = d.dropna(subset=["c_m1", "c_p1"])
    d = d[d["c_m1"] > 0]
    d["ann_ret_2d"] = d["c_p1"] / d["c_m1"] - 1.0
    d["avail_date"] = d["ap1"]

    out = d[["ticker", "avail_date", "ann_day", "ann_ret_2d"]].copy()
    out = out.sort_values(["ticker", "avail_date"],
                          kind="mergesort").reset_index(drop=True)
    logger.info(
        f"C27 ann_ret_2d: {len(out)} events, {out['ticker'].nunique()} tickers "
        f"(from {len(ev)} deduped Item-2.02 8-Ks; "
        f"{len(ev) - int(valid.sum())} at calendar edges, "
        f"{int(valid.sum()) - len(out)} missing a close)")
    if write:
        out.to_parquet(CACHE_DIR / C27_NAME, index=False)
        logger.info(f"Saved {len(out)} rows to {CACHE_DIR / C27_NAME}")
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Build C25/C26/C27 candidate caches")
    p.add_argument("--only", choices=["c25", "c26", "c27"], default=None,
                   help="build only one signal (default: all three)")
    args = p.parse_args()
    if args.only in (None, "c25"):
        compute_net_issuance(write=True)
    if args.only in (None, "c26"):
        compute_asset_growth(write=True)
    if args.only in (None, "c27"):
        compute_ann_ret_2d(write=True)


if __name__ == "__main__":
    main()
