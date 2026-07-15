"""C25-C29: fundamentals-family candidates on data already in hand.

C25/C26/C27 registered 2026-07-15 (reports/factor_preregistration.md row
"C25/C26/C27 registration", commit `28b84c6`; FACTORS.md C25/C26/C27).
C28/C29 registered 2026-07-15 (row "C28/C29 registration", commit `528404c`;
FACTORS.md C28/C29). Family=XS-IC, roles=sel, ONE primary v0 look each, bar
sn incr |t| >= 1.5 -> candidate else rejected. Each signal writes its own
per-signal cache and merges as a filing-date FactorSource (availability = the
stamped date, v0 t+1 lag applied at the panel merge, like every other
filing-dated factor).

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

- **C28 `accruals_sloan`** (Sloan 1996; high accruals -> low returns). Total
  accruals = (NI_ttm - CFO_ttm) / avg total Assets, all as-filed first-filed.
  NI_ttm reuses `fundamentals_pit.net_income_ttm_events` (qtrs=1 TTM with the
  sue_pead Q4-derivation discipline). CFO from
  `NetCashProvidedByUsedInOperatingActivities`: the cash-flow statement is
  reported CUMULATIVE YTD (qtrs=1 = Q1 only, qtrs=2 = H1, qtrs=3 = 9mo,
  qtrs=4 = FY — inspected at build, ~1 qtrs=1 row per fiscal year), so the
  quarterly CFO is qtrs=1 directly for Q1 and YTD-DIFFERENCED (qtrs=k minus
  qtrs=(k-1) one quarter earlier) for Q2/Q3/Q4; ~25% direct, ~75% differenced
  (logged). CFO_ttm = trailing 4 quarterly-CFO sum (span <= 320d). Assets =
  qtrs=0 instants, avg of the TTM window's start (~365+/-45d earlier) and end.
  Availability = max of every constituent filed date (NI TTM, CFO TTM, both
  Assets instants). Collapsed to as-of states (`fundamentals_pit.as_of_states`)
  so a delinquent old-period filing never rolls the value backwards.

- **C29 `gross_profitability`** (Novy-Marx 2013; high GP/A -> high returns).
  GP_ttm / total Assets. Quarterly gross profit per (cik, quarter) by priority:
  `GrossProfit` where tagged, else `Revenues - CostOfRevenue`, else
  `Revenues - CostOfGoodsAndServicesSold` (income-statement items, quarterly
  via `sue_pead.quarterly_eps` with derived Q4; source mix logged). GP_ttm =
  trailing 4 quarterly-GP sum (span <= 320d). Assets = qtrs=0 instant level at
  period_end. Availability = max constituent filed date; same as-of collapse.
  GrossProfit is thinly tagged (191/497 ciks); the fallback lifts computable
  coverage to ~62% of ciks — the builder and the ledger state the thinness.

Usage:
    python -m alpha_graph.signals.fundamentals_candidates          # all five
    python -m alpha_graph.signals.fundamentals_candidates --only c28
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR
from alpha_graph.data.shares_pit import CAP_UNIT_MISMATCH_TICKERS
from alpha_graph.signals.fundamentals_pit import (
    TTM_MAX_SPAN_DAYS,
    TTM_QUARTERS,
    as_of_states,
    net_income_ttm_events,
)
from alpha_graph.signals.sue_pead import first_filed, quarterly_eps

SEASONAL_DAYS = 365          # target gap to the observation ~1 year earlier
SEASONAL_TOL_DAYS = 45       # match tolerance around SEASONAL_DAYS (like SUE)
SPLIT_UP = 1.9               # consecutive count ratio above this = split-like
SPLIT_DOWN = 0.55            # ... or below this (reverse split / artifact)
ACCEPT_CUTOFF_HOUR = 16      # ET; accepted at/after this -> next-trading-day a
ITEM_202 = "2.02"

MONEY_UOM = "USD"
CFO_TAG = "NetCashProvidedByUsedInOperatingActivities"
ASSETS_TAG = "Assets"
GP_TAG = "GrossProfit"
REV_TAG = "Revenues"
COR_TAG = "CostOfRevenue"
COGS_TAG = "CostOfGoodsAndServicesSold"
QTR_GAP_DAYS = 91            # one fiscal quarter (13 weeks); YTD-diff match gap
QTR_GAP_TOL = 31             # window [60, 122]d isolates one quarter from two

C25_NAME = "net_issuance_12m.parquet"
C26_NAME = "asset_growth_yoy.parquet"
C27_NAME = "ann_ret_2d.parquet"
C28_NAME = "accruals_sloan.parquet"
C29_NAME = "gross_profitability.parquet"


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
# Shared: cumulative-YTD -> quarterly, TTM sums, Assets instants
# --------------------------------------------------------------------------- #

def _prev_quarter_match(dk: np.ndarray, dkm1: np.ndarray) -> np.ndarray:
    """For each ddate in `dk`, the index in `dkm1` of the observation ~one
    fiscal quarter (QTR_GAP_DAYS) earlier, within [+/-QTR_GAP_TOL] and nearest
    to the target; -1 if none. `dkm1` must be sorted ascending.

    The [60, 122]d window isolates the immediately preceding cumulative period
    (~91d) from two quarters back (~182d): a missing prior quarter yields no
    match rather than a wrong-gap difference."""
    dk = dk.astype("datetime64[ns]")
    dkm1 = dkm1.astype("datetime64[ns]")
    lo_gap = np.timedelta64(QTR_GAP_DAYS + QTR_GAP_TOL, "D")
    hi_gap = np.timedelta64(QTR_GAP_DAYS - QTR_GAP_TOL, "D")
    target = np.timedelta64(QTR_GAP_DAYS, "D")
    lo = np.searchsorted(dkm1, dk - lo_gap, side="left")
    hi = np.searchsorted(dkm1, dk - hi_gap, side="right")
    out = np.full(len(dk), -1)
    for i in range(len(dk)):
        if hi[i] <= lo[i]:
            continue
        cand = np.arange(lo[i], hi[i])
        out[i] = cand[np.argmin(np.abs((dk[i] - dkm1[cand]) - target))]
    return out


def quarterly_cfo(first_cfo: pd.DataFrame) -> pd.DataFrame:
    """Quarterly operating cash flow from first-filed cumulative-YTD rows.

    Input: `first_filed` output for the CFO tag (one row per (cik, ddate,
    qtrs)). The cash-flow statement reports YTD, so quarterly CFO is qtrs=1
    directly (Q1, PREFERRED where present) and YTD-DIFFERENCED elsewhere —
    value(qtrs=k) - value(qtrs=k-1) at the fiscal quarter ~QTR_GAP_DAYS
    earlier. Availability of a differenced quarter = the max of the two
    constituent filings' filed dates.

    Returns cik, ticker, ddate (quarter end), cfo_q, avail, source
    ('q1' | 'diff'). Logs the direct/differenced mix (the registration's
    "which form dominates" requirement)."""
    f = first_cfo[first_cfo["qtrs"].isin([1, 2, 3, 4])].copy()
    f["ddate"] = pd.to_datetime(f["ddate"])
    f["filed"] = pd.to_datetime(f["filed"])
    parts, n_direct, n_diff, n_unmatched = [], 0, 0, 0
    for _, g in f.groupby("cik", sort=False):
        g = g.sort_values(["ddate", "qtrs"])
        rows = []
        lvl = {k: g[g["qtrs"] == k] for k in (1, 2, 3, 4)}
        for r in lvl[1].itertuples():                    # Q1 direct (prefer q1)
            rows.append((r.cik, r.ticker, r.ddate, r.value, r.filed, "q1"))
            n_direct += 1
        for k in (2, 3, 4):                              # YTD differencing
            gk, gkm1 = lvl[k], lvl[k - 1].sort_values("ddate")
            if gk.empty:
                continue
            if gkm1.empty:
                n_unmatched += len(gk)
                continue
            dk = gk["ddate"].to_numpy()
            m = _prev_quarter_match(dk, gkm1["ddate"].to_numpy())
            vk, fk, tk, ck = (gk["value"].to_numpy(), gk["filed"].to_numpy(),
                              gk["ticker"].to_numpy(), gk["cik"].to_numpy())
            vkm1 = gkm1["value"].to_numpy()
            fkm1 = gkm1["filed"].to_numpy()
            for i in range(len(gk)):
                j = m[i]
                if j < 0:
                    n_unmatched += 1
                    continue
                rows.append((ck[i], tk[i], pd.Timestamp(dk[i]),
                             vk[i] - vkm1[j],
                             max(pd.Timestamp(fk[i]), pd.Timestamp(fkm1[j])),
                             "diff"))
                n_diff += 1
        if rows:
            parts.append(pd.DataFrame(
                rows, columns=["cik", "ticker", "ddate", "cfo_q", "avail",
                               "source"]))
    if not parts:
        return pd.DataFrame(columns=["cik", "ticker", "ddate", "cfo_q",
                                     "avail", "source"])
    out = pd.concat(parts, ignore_index=True)
    # prefer qtrs=1 on the rare (cik, ddate) reported both standalone and YTD
    out["_prio"] = (out["source"] != "q1").astype(int)
    out = out.sort_values(["cik", "ddate", "_prio"], kind="mergesort")
    out = out.drop_duplicates(["cik", "ddate"], keep="first").drop(columns="_prio")
    tot = n_direct + n_diff
    logger.info(
        f"C28 quarterly CFO: {tot} quarters "
        f"({n_direct} qtrs=1-direct {100 * n_direct / max(tot, 1):.1f}%, "
        f"{n_diff} YTD-differenced {100 * n_diff / max(tot, 1):.1f}%), "
        f"{n_unmatched} dropped (no prior cumulative quarter)")
    return out.reset_index(drop=True)


def _ttm(q: pd.DataFrame, val_col: str, avail_col: str,
         ddate_col: str = "ddate", cik_col: str = "cik") -> pd.DataFrame:
    """Trailing-4-quarter sum of `val_col`, keyed per cik and ordered by
    `ddate_col`. Mirrors `fundamentals_pit.net_income_ttm_events`: the window
    must span <= TTM_MAX_SPAN_DAYS (a missing quarter pushes it to ~364d ->
    NaN), and the TTM availability = max of the 4 constituent quarters'
    `avail_col`. Adds `ttm` and `ttm_avail`; drops undefined windows."""
    q = q.drop_duplicates([cik_col, ddate_col], keep="first")
    parts = []
    for _, g in q.groupby(cik_col, sort=False):
        g = g.sort_values(ddate_col).reset_index(drop=True)
        g["ttm"] = g[val_col].rolling(TTM_QUARTERS).sum()
        span = (g[ddate_col] - g[ddate_col].shift(TTM_QUARTERS - 1)).dt.days
        g["ttm"] = g["ttm"].where(span <= TTM_MAX_SPAN_DAYS)
        av = pd.concat([g[avail_col].shift(k) for k in range(TTM_QUARTERS)],
                       axis=1)
        g["ttm_avail"] = av.max(axis=1, skipna=False)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True) if parts else q.assign(
        ttm=np.nan, ttm_avail=pd.NaT)
    return out.dropna(subset=["ttm", "ttm_avail"])


def assets_instant(facts: pd.DataFrame) -> pd.DataFrame:
    """First-filed total Assets, qtrs=0 instants: cik, ticker, period_end,
    assets, assets_filed."""
    a = first_filed(facts, ASSETS_TAG, MONEY_UOM)
    a = a[a["qtrs"] == 0].copy()
    a["period_end"] = pd.to_datetime(a["ddate"])
    a["assets_filed"] = pd.to_datetime(a["filed"])
    return a[["cik", "ticker", "period_end", "value", "assets_filed"]].rename(
        columns={"value": "assets"})


def avg_assets(facts: pd.DataFrame) -> pd.DataFrame:
    """Average of total Assets at the TTM window's end and start (~365+/-45d
    earlier), the Sloan denominator. Returns cik, period_end, avg_assets,
    assets_avail (= max of the two instants' filed dates). Rows without a
    ~1-year-earlier instant are dropped (no beginning-of-year balance)."""
    ai = assets_instant(facts)
    parts = []
    for _, g in ai.groupby("cik", sort=False):
        g = g.sort_values("period_end").reset_index(drop=True)
        dd = g["period_end"].to_numpy().astype("datetime64[ns]")
        val = g["assets"].to_numpy(dtype=float)
        filed = g["assets_filed"].to_numpy().astype("datetime64[ns]")
        match = _yoy_match(dd)
        avg = np.full(len(g), np.nan)
        av = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")
        for i in range(len(g)):
            j = match[i]
            if j < 0:
                continue
            avg[i] = (val[i] + val[j]) / 2.0
            av[i] = max(filed[i], filed[j])
        parts.append(pd.DataFrame({"cik": g["cik"].to_numpy(), "period_end": dd,
                                   "avg_assets": avg, "assets_avail": av}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["cik", "period_end", "avg_assets", "assets_avail"])
    return out.dropna(subset=["avg_assets"])


def quarterly_gross_profit(facts: pd.DataFrame) -> pd.DataFrame:
    """Quarterly gross profit per (cik, quarter) with the pinned fallback
    priority: GrossProfit where tagged, else Revenues - CostOfRevenue, else
    Revenues - CostOfGoodsAndServicesSold. Income-statement items -> quarterly
    via `sue_pead.quarterly_eps` (derived Q4 discipline). A fallback quarter's
    availability = max of its constituent filings' filed dates.

    Returns cik, ddate (quarter end), gross_profit, gp_avail, gp_source. Logs
    the source mix (the registration's reported-mix requirement)."""
    def q(tag: str, name: str) -> pd.DataFrame:
        f = first_filed(facts, tag, MONEY_UOM)
        if f.empty:
            return pd.DataFrame(columns=["cik", "ddate", name, name + "_f"])
        qq = quarterly_eps(f)
        qq["ddate"] = pd.to_datetime(qq["ddate"])
        qq["stmt_filed"] = pd.to_datetime(qq["stmt_filed"])
        return qq[["cik", "ddate", "eps", "stmt_filed"]].rename(
            columns={"eps": name, "stmt_filed": name + "_f"})

    m = q(GP_TAG, "gp")
    for tag, name in [(REV_TAG, "rev"), (COR_TAG, "cor"), (COGS_TAG, "cogs")]:
        m = m.merge(q(tag, name), on=["cik", "ddate"], how="outer")
    m = m.sort_values(["cik", "ddate"], kind="mergesort").reset_index(drop=True)

    gp = np.full(len(m), np.nan)
    avail = np.full(len(m), np.datetime64("NaT"), dtype="datetime64[ns]")
    source = np.array([None] * len(m), dtype=object)
    have = {c: m[c].notna().to_numpy() for c in ("gp", "rev", "cor", "cogs")}
    gpv, revv, corv, cogsv = (m["gp"].to_numpy(dtype=float),
                              m["rev"].to_numpy(dtype=float),
                              m["cor"].to_numpy(dtype=float),
                              m["cogs"].to_numpy(dtype=float))
    gpf = pd.to_datetime(m["gp_f"]).to_numpy()
    revf = pd.to_datetime(m["rev_f"]).to_numpy()
    corf = pd.to_datetime(m["cor_f"]).to_numpy()
    cogsf = pd.to_datetime(m["cogs_f"]).to_numpy()
    for i in range(len(m)):
        if have["gp"][i]:
            gp[i], avail[i], source[i] = gpv[i], gpf[i], "GrossProfit"
        elif have["rev"][i] and have["cor"][i]:
            gp[i] = revv[i] - corv[i]
            avail[i] = max(revf[i], corf[i])
            source[i] = "Revenues-CostOfRevenue"
        elif have["rev"][i] and have["cogs"][i]:
            gp[i] = revv[i] - cogsv[i]
            avail[i] = max(revf[i], cogsf[i])
            source[i] = "Revenues-COGS"
    out = pd.DataFrame({"cik": m["cik"].to_numpy(), "ddate": m["ddate"].to_numpy(),
                        "gross_profit": gp, "gp_avail": avail,
                        "gp_source": source})
    out = out.dropna(subset=["gross_profit"]).reset_index(drop=True)
    mix = out["gp_source"].value_counts()
    logger.info(
        f"C29 quarterly gross profit: {len(out)} quarters; source mix "
        + ", ".join(f"{s} {100 * n / max(len(out), 1):.1f}%"
                    for s, n in mix.items()))
    return out


# --------------------------------------------------------------------------- #
# C28 accruals_sloan
# --------------------------------------------------------------------------- #

def compute_accruals(facts: pd.DataFrame | None = None,
                     write: bool = False) -> pd.DataFrame:
    """Build the C28 cache: ticker, avail_date, accruals_sloan = (NI_ttm -
    CFO_ttm) / avg total Assets. Injectable `facts` for tests."""
    if facts is None:
        facts = pd.read_parquet(CACHE_DIR / "xbrl_facts.parquet")

    ni = net_income_ttm_events(facts)[
        ["cik", "ticker", "period_end", "net_income_ttm", "avail_date"]]
    cfo = _ttm(quarterly_cfo(first_filed(facts, CFO_TAG, MONEY_UOM)),
               "cfo_q", "avail").rename(
        columns={"ddate": "period_end", "ttm": "cfo_ttm",
                 "ttm_avail": "cfo_avail"})[
        ["cik", "period_end", "cfo_ttm", "cfo_avail"]]
    av = avg_assets(facts)

    m = ni.merge(cfo, on=["cik", "period_end"], how="inner").merge(
        av, on=["cik", "period_end"], how="inner")
    m = m[m["avg_assets"] > 0].copy()
    m["accruals_sloan"] = (m["net_income_ttm"] - m["cfo_ttm"]) / m["avg_assets"]
    for c in ("avail_date", "cfo_avail", "assets_avail"):
        m[c] = pd.to_datetime(m[c])
    m["avail_date"] = m[["avail_date", "cfo_avail", "assets_avail"]].max(axis=1)

    out = as_of_states(
        m[["ticker", "avail_date", "period_end", "accruals_sloan"]],
        "accruals_sloan")
    out = out.sort_values(["ticker", "avail_date"],
                          kind="mergesort").reset_index(drop=True)
    logger.info(
        f"C28 accruals_sloan: {len(out)} state rows, "
        f"{out['ticker'].nunique()} tickers (from {len(m)} scored quarters)")
    if write:
        out.to_parquet(CACHE_DIR / C28_NAME, index=False)
        logger.info(f"Saved {len(out)} rows to {CACHE_DIR / C28_NAME}")
    return out


# --------------------------------------------------------------------------- #
# C29 gross_profitability
# --------------------------------------------------------------------------- #

def compute_gross_profitability(facts: pd.DataFrame | None = None,
                                write: bool = False) -> pd.DataFrame:
    """Build the C29 cache: ticker, avail_date, gross_profitability = GP_ttm /
    total Assets. Injectable `facts` for tests."""
    if facts is None:
        facts = pd.read_parquet(CACHE_DIR / "xbrl_facts.parquet")

    gp = _ttm(quarterly_gross_profit(facts), "gross_profit", "gp_avail").rename(
        columns={"ddate": "period_end", "ttm": "gp_ttm",
                 "ttm_avail": "gp_ttm_avail"})[
        ["cik", "period_end", "gp_ttm", "gp_ttm_avail"]]
    ai = assets_instant(facts)[["cik", "ticker", "period_end", "assets",
                                "assets_filed"]]

    m = gp.merge(ai, on=["cik", "period_end"], how="inner")
    m = m[m["assets"] > 0].copy()
    m["gross_profitability"] = m["gp_ttm"] / m["assets"]
    for c in ("gp_ttm_avail", "assets_filed"):
        m[c] = pd.to_datetime(m[c])
    m["avail_date"] = m[["gp_ttm_avail", "assets_filed"]].max(axis=1)

    out = as_of_states(
        m[["ticker", "avail_date", "period_end", "gross_profitability"]],
        "gross_profitability")
    out = out.sort_values(["ticker", "avail_date"],
                          kind="mergesort").reset_index(drop=True)
    logger.info(
        f"C29 gross_profitability: {len(out)} state rows, "
        f"{out['ticker'].nunique()} tickers (from {len(m)} scored quarters)")
    if write:
        out.to_parquet(CACHE_DIR / C29_NAME, index=False)
        logger.info(f"Saved {len(out)} rows to {CACHE_DIR / C29_NAME}")
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Build C25-C29 candidate caches")
    p.add_argument("--only",
                   choices=["c25", "c26", "c27", "c28", "c29"], default=None,
                   help="build only one signal (default: all five)")
    args = p.parse_args()
    if args.only in (None, "c25"):
        compute_net_issuance(write=True)
    if args.only in (None, "c26"):
        compute_asset_growth(write=True)
    if args.only in (None, "c27"):
        compute_ann_ret_2d(write=True)
    if args.only in (None, "c28"):
        compute_accruals(write=True)
    if args.only in (None, "c29"):
        compute_gross_profitability(write=True)


if __name__ == "__main__":
    main()
