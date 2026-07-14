"""C20 `insider_cluster_concurrent`: concurrency-conditioned insider clusters.

Registered 2026-07-14 (FACTORS.md C20; reports/factor_preregistration.md),
6 looks and the success bar pinned in the ledger PRE-computation. This module
is a thin layer over the C16 machinery: it selects the qualifying subset of
the cached C16 event list (`insider_cluster_c16_events.parquet`) and reuses
`insider_cluster.build_series` unchanged — corrected timing (credit_offset=2:
entry close(d+1), first credit close(d+1)->close(d+2)), 63td hold, EW,
empty days = 0.0 cash and kept in the regression.

Qualification (the registered rule, exact semantics):

- concurrency count c(e) = the number of C16 events on the FULL event list
  (all tickers, tradeable or not, the event itself and same-day events
  included) with event_date in (d - 21 calendar days, d], d = e's fire
  date — half-open: an event exactly 21 days earlier does NOT count.
- e QUALIFIES iff c(e) >= the 80th percentile (np.percentile, linear
  interpolation) of {c(e') : event_date(e') in (d - 3y, d)} — the same
  statistic over STRICTLY-prior events in the trailing 3 years
  (pd.DateOffset(years=3); both interval ends open). Each prior count
  c(e') depends only on events at or before e' < d, so the threshold is
  PIT: adding future events changes no earlier qualification.
- WARMUP (build-time pin, ledgered 2026-07-14, pre-evaluation; not a
  look): with fewer than MIN_PRIOR_3Y = 30 prior events in the 3y window
  the event does NOT qualify (the percentile is not estimable early).
  n_prior == 0 with min_prior == 0 (tests only) qualifies vacuously.

Caches:
- data/cache/insider_cluster_c20_events.parquet — the qualifying subset
  with concurrency stats (conc_count, n_prior_3y, pctl_thresh) and the
  tradeable flag build_series assigns.
- data/cache/insider_cluster_c20.parquet — daily series: ret, n_active,
  n_active_events, on the full market calendar.

Evaluation — EXACTLY the 6 pinned looks, printed by main():

1.   FF5+MOM attribution (`ff_attribution`, excess=False — long-only total
     returns) on the full C20 series.
2-3. The same regression on each half, split at the series midpoint DATE.
4.   Within-ticker date-permutation placebo: the tradeable qualifying
     events' dates are permuted within ticker (uniform, without
     replacement, over that ticker's price-coverage days — the C16
     investigation's convention), the book rebuilt and regressed
     identically; median over seeds {0, 1, 2}.
5.   Same-ticker -252td matched control per the C16-investigation
     conventions: control = same ticker at pos(event) - 252 on the market
     calendar; skip (in this order) if the control position is off the
     calendar, the ticker has no close at the control date, or the control
     position is within +/-63td of ANY of that ticker's tradeable C16
     events (the parent list, not just qualifying ones); match rate
     reported. Statistic = Newey-West HAC t (lags 32, ff_attribution's
     convention) of the mean of the paired daily diff real - control,
     both books rebuilt on the matched subset over the full calendar.
6.   Same-window NON-event control (the decisive look): for each tradeable
     qualifying event, draw (one np.random.default_rng(0), events in
     (event_date, ticker) order) a uniform random panel name with non-NaN
     closes over the identical [pos0+1, pos0+64] trading-day window
     (clipped at the calendar end exactly like the real credited window)
     and NO tradeable C16 event within +/-63td of pos0; the drawn names
     are held through the identical windows with the same build_series
     machinery and regressed FF5+MOM (excess=False). ALSO reported: HAC t
     (lags 32) of the mean of the paired daily diff real - nonevent —
     the pin's decisive number (bar condition 3).

Success bar (pinned pre-computation, applied mechanically): full alpha
HAC t >= +2.0 AND (real alpha - permutation-median alpha) >= +3%/yr AND
paired (real - same-window-control) HAC t >= +2.0 -> candidate; ANY miss
-> rejected.

Usage:
    python -m alpha_graph.signals.insider_cluster_concurrent
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR
from alpha_graph.eval.ff_attribution import (
    DEFAULT_HAC_LAGS,
    ff_attribution,
    load_factors,
)
from alpha_graph.eval.ic_tools import hac_tstat
from alpha_graph.signals.insider_cluster import (
    CREDIT_OFFSET,
    HOLD_TDAYS,
    build_series,
    pinned_looks,
)

CONC_WINDOW_CAL_DAYS = 21    # trailing market-wide count window (calendar days)
CONC_LOOKBACK_YEARS = 3      # PIT percentile lookback (DateOffset years)
CONC_PCTL = 80.0             # qualification percentile
MIN_PRIOR_3Y = 30            # warmup pin: fewer prior events -> not qualified

CTRL_SHIFT_TD = 252          # look 5: same-ticker control offset (trading days)
COLLIDE_TD = 63              # looks 5-6: +/- collision radius (trading days)
PERM_SEEDS = (0, 1, 2)       # look 4 seeds
NONEVENT_SEED = 0            # look 6 seed
PAIRED_HAC_LAGS = DEFAULT_HAC_LAGS   # 32 — same NW convention as ff_attribution

FULL_T_BAR = 2.0             # bar 1: full-sample alpha HAC t
PERM_GAP_BAR = 0.03          # bar 2: real - permutation-median alpha (ann.)
PAIRED_T_BAR = 2.0           # bar 3: paired (real - nonevent) HAC t

SERIES_CACHE = "insider_cluster_c20.parquet"
EVENTS_CACHE = "insider_cluster_c20_events.parquet"
C16_EVENTS_CACHE = "insider_cluster_c16_events.parquet"


# --------------------------------------------------------------------------- #
# Qualification
# --------------------------------------------------------------------------- #

def concurrency_stats(events: pd.DataFrame,
                      window_days: int = CONC_WINDOW_CAL_DAYS,
                      lookback_years: int = CONC_LOOKBACK_YEARS,
                      pctl: float = CONC_PCTL,
                      min_prior: int = MIN_PRIOR_3Y) -> pd.DataFrame:
    """Attach conc_count, n_prior_3y, pctl_thresh, qualifies to every event.

    conc_count(e) counts FULL-list events with event_date in (d-window, d]
    (self and same-day events included; exactly `window_days` earlier
    excluded). The threshold is the `pctl` percentile of the strictly-prior
    events' own counts within (d - lookback_years, d), NaN (and qualifies
    False) while fewer than `min_prior` prior events exist — the warmup
    build-time pin.
    """
    ev = events.copy()
    dates = pd.DatetimeIndex(ev["event_date"])
    darr = dates.values
    dsort = np.sort(darr)
    win = np.timedelta64(window_days, "D")

    conc = (np.searchsorted(dsort, darr, side="right")
            - np.searchsorted(dsort, darr - win, side="right")).astype(int)
    conc_sorted = (np.searchsorted(dsort, dsort, side="right")
                   - np.searchsorted(dsort, dsort - win, side="right")
                   ).astype(int)

    cutoff = (dates - pd.DateOffset(years=lookback_years)).values
    lo = np.searchsorted(dsort, cutoff, side="right")   # date > d - 3y
    hi = np.searchsorted(dsort, darr, side="left")      # date < d
    n_prior = (hi - lo).astype(int)

    thresh = np.full(len(ev), np.nan)
    qual = np.zeros(len(ev), dtype=bool)
    for i in range(len(ev)):
        if n_prior[i] < min_prior:
            continue
        if n_prior[i] == 0:            # reachable only when min_prior == 0
            thresh[i] = -np.inf
            qual[i] = True
            continue
        t = float(np.percentile(conc_sorted[lo[i]:hi[i]], pctl))
        thresh[i] = t
        qual[i] = bool(conc[i] >= t)

    ev["conc_count"] = conc
    ev["n_prior_3y"] = n_prior
    ev["pctl_thresh"] = thresh
    ev["qualifies"] = qual
    return ev


def build_c20(events: pd.DataFrame, market: pd.DataFrame,
              window_days: int = CONC_WINDOW_CAL_DAYS,
              lookback_years: int = CONC_LOOKBACK_YEARS,
              pctl: float = CONC_PCTL,
              min_prior: int = MIN_PRIOR_3Y,
              hold_days: int = HOLD_TDAYS,
              credit_offset: int = CREDIT_OFFSET,
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """C16 events + market -> (daily C20 series, qualifying events, stats).

    The portfolio construction is `insider_cluster.build_series` verbatim
    (corrected timing by default) on the qualifying subset.
    """
    stats = concurrency_stats(events, window_days=window_days,
                              lookback_years=lookback_years, pctl=pctl,
                              min_prior=min_prior)
    qual = stats.loc[stats["qualifies"]].drop(columns=["qualifies"])
    qual = qual.reset_index(drop=True)
    series, qual = build_series(qual, market, hold_days=hold_days,
                                credit_offset=credit_offset)
    return series, qual, stats


# --------------------------------------------------------------------------- #
# Look 4: within-ticker date-permutation placebo (median of seeds {0,1,2})
# --------------------------------------------------------------------------- #

def permutation_look(qual_events: pd.DataFrame, market: pd.DataFrame,
                     factors: pd.DataFrame | None = None,
                     seeds: tuple[int, ...] = PERM_SEEDS) -> dict:
    """Permute the tradeable qualifying events' dates within ticker
    (uniform, without replacement, over the ticker's price-coverage days —
    the C16 investigation's convention), rebuild, regress; median of seeds."""
    ev = qual_events.loc[qual_events["tradeable"]]
    px = market.pivot(index="date", columns="ticker",
                      values="close").sort_index()
    cov = {tk: px.index[px[tk].notna().to_numpy()] for tk in px.columns}
    counts = ev["ticker"].value_counts().sort_index()

    per_seed = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rows = []
        for tk, k in counts.items():
            days = cov.get(tk)
            if days is None or len(days) == 0:
                continue
            pick = rng.choice(len(days), size=min(int(k), len(days)),
                              replace=False)
            rows.extend((tk, d) for d in days[np.sort(pick)])
        pev = pd.DataFrame(rows, columns=["ticker", "event_date"])
        series, _ = build_series(pev, market)
        res = ff_attribution(series["ret"], factors=factors, excess=False)
        per_seed.append({"seed": seed, "n_events": len(pev),
                         "alpha_ann": res["alpha_ann"],
                         "alpha_t_hac": res["alpha_t_hac"]})
    return {"per_seed": per_seed,
            "alpha_ann_median": float(np.median(
                [r["alpha_ann"] for r in per_seed])),
            "alpha_t_hac_median": float(np.median(
                [r["alpha_t_hac"] for r in per_seed]))}


# --------------------------------------------------------------------------- #
# Look 5: same-ticker -252td matched control, paired daily diff
# --------------------------------------------------------------------------- #

def matched_control_look(qual_events: pd.DataFrame, c16_events: pd.DataFrame,
                         market: pd.DataFrame,
                         shift_td: int = CTRL_SHIFT_TD,
                         collide_td: int = COLLIDE_TD,
                         lags: int = PAIRED_HAC_LAGS) -> dict:
    """Same ticker at pos(event) - shift_td; skip per the C16-investigation
    conventions (calendar edge / no price at the control date / within
    +/-collide_td of ANY of the ticker's tradeable C16 events). Statistic:
    HAC t of the mean of the paired daily diff real - control."""
    px = market.pivot(index="date", columns="ticker",
                      values="close").sort_index()
    cal = px.index
    real = (qual_events.loc[qual_events["tradeable"]]
            .sort_values(["event_date", "ticker"]).reset_index(drop=True))
    c16t = c16_events.loc[c16_events["tradeable"]]
    pos_by_ticker = {
        tk: cal.searchsorted(sub["event_date"].values, side="left")
        for tk, sub in c16t.groupby("ticker")}

    pos = cal.searchsorted(real["event_date"].values, side="left")
    rows_real, rows_ctrl = [], []
    n_edge = n_px = n_collide = 0
    for i, row in enumerate(real.itertuples()):
        cpos = int(pos[i]) - shift_td
        if cpos < 0:
            n_edge += 1
            continue
        cdate = cal[cpos]
        if row.ticker not in px.columns or pd.isna(px.at[cdate, row.ticker]):
            n_px += 1
            continue
        if np.abs(pos_by_ticker[row.ticker] - cpos).min() <= collide_td:
            n_collide += 1
            continue
        rows_real.append((row.ticker, row.event_date))
        rows_ctrl.append((row.ticker, cdate))

    real_ev = pd.DataFrame(rows_real, columns=["ticker", "event_date"])
    ctrl_ev = pd.DataFrame(rows_ctrl, columns=["ticker", "event_date"])
    s_real, _ = build_series(real_ev, market)
    s_ctrl, _ = build_series(ctrl_ev, market)
    diff = s_real["ret"] - s_ctrl["ret"]
    return {"n_events": len(real), "n_matched": len(real_ev),
            "match_rate": len(real_ev) / len(real) if len(real) else np.nan,
            "skips": {"edge": n_edge, "px": n_px, "collide": n_collide},
            "diff_mean_ann": float(diff.mean() * 252.0),
            "diff_t_hac": float(hac_tstat(diff.to_numpy(), lags=lags)),
            "n_days": int(diff.notna().sum())}


# --------------------------------------------------------------------------- #
# Look 6: same-window non-event control (the decisive look)
# --------------------------------------------------------------------------- #

def nonevent_control_look(qual_events: pd.DataFrame,
                          c16_events: pd.DataFrame, market: pd.DataFrame,
                          factors: pd.DataFrame | None = None,
                          seed: int = NONEVENT_SEED,
                          collide_td: int = COLLIDE_TD,
                          hold_days: int = HOLD_TDAYS,
                          credit_offset: int = CREDIT_OFFSET,
                          lags: int = PAIRED_HAC_LAGS) -> dict:
    """Count-matched random panel names through the identical windows.

    Per tradeable qualifying event at pos0: eligible names have non-NaN
    closes on every trading day of [pos0+1, min(pos0 + credit_offset +
    hold_days - 1, end)] (the entry close through the last credited close,
    clipped like the real window) and no tradeable C16 event within
    +/-collide_td of pos0. One uniform draw per event, single rng(seed),
    events in (event_date, ticker) order. Control book = build_series on
    the drawn (name, same event_date) list; FF5+MOM (excess=False) on it;
    plus HAC t of the mean of the paired daily diff real - control."""
    px = market.pivot(index="date", columns="ticker",
                      values="close").sort_index()
    cal = px.index
    n = len(cal)
    valid = px.notna().to_numpy()
    tickers = px.columns.to_numpy()
    tick_idx = {tk: j for j, tk in enumerate(px.columns)}

    near = np.zeros((n, len(tickers)), dtype=bool)
    c16t = c16_events.loc[c16_events["tradeable"]]
    c16_pos = cal.searchsorted(c16t["event_date"].values, side="left")
    for p, tk in zip(c16_pos, c16t["ticker"]):
        j = tick_idx.get(tk)
        if j is not None:
            near[max(0, int(p) - collide_td):
                 min(n, int(p) + collide_td + 1), j] = True

    real = (qual_events.loc[qual_events["tradeable"]]
            .sort_values(["event_date", "ticker"]).reset_index(drop=True))
    pos = cal.searchsorted(real["event_date"].values, side="left")
    rng = np.random.default_rng(seed)
    rows_real, rows_ctrl = [], []
    n_unmatched = 0
    for i, row in enumerate(real.itertuples()):
        pos0 = int(pos[i])
        w0 = pos0 + 1
        w1 = min(pos0 + credit_offset + hold_days - 1, n - 1)
        if w0 > n - 1:
            n_unmatched += 1
            continue
        ok = valid[w0:w1 + 1].all(axis=0) & ~near[pos0]
        cand = np.flatnonzero(ok)
        if len(cand) == 0:
            n_unmatched += 1
            continue
        j = int(rng.choice(cand))
        rows_real.append((row.ticker, row.event_date))
        rows_ctrl.append((tickers[j], row.event_date))

    real_ev = pd.DataFrame(rows_real, columns=["ticker", "event_date"])
    ctrl_ev = pd.DataFrame(rows_ctrl, columns=["ticker", "event_date"])
    s_real, _ = build_series(real_ev, market, hold_days=hold_days,
                             credit_offset=credit_offset)
    s_ctrl, _ = build_series(ctrl_ev, market, hold_days=hold_days,
                             credit_offset=credit_offset)
    res_ctrl = ff_attribution(s_ctrl["ret"], factors=factors, excess=False)
    diff = s_real["ret"] - s_ctrl["ret"]
    return {"n_events": len(real), "n_matched": len(real_ev),
            "n_unmatched": n_unmatched,
            "n_ctrl_names": int(ctrl_ev["ticker"].nunique()),
            "ctrl_alpha_ann": res_ctrl["alpha_ann"],
            "ctrl_alpha_t_hac": res_ctrl["alpha_t_hac"],
            "ctrl_mkt_beta": res_ctrl["loadings"]["mkt_rf"],
            "ctrl_r2": res_ctrl["r2"],
            "diff_mean_ann": float(diff.mean() * 252.0),
            "diff_t_hac": float(hac_tstat(diff.to_numpy(), lags=lags)),
            "n_days": int(diff.notna().sum())}


# --------------------------------------------------------------------------- #
# The pinned bar
# --------------------------------------------------------------------------- #

def decide(full_look: dict, perm: dict, nonevent: dict
           ) -> tuple[str, dict[str, bool]]:
    """Mechanical application of the pinned bar; ANY miss -> rejected."""
    conds = {
        "full_t": full_look["alpha_t_hac"] >= FULL_T_BAR,
        "perm_gap": (full_look["alpha_ann"]
                     - perm["alpha_ann_median"]) >= PERM_GAP_BAR,
        "paired_t": nonevent["diff_t_hac"] >= PAIRED_T_BAR,
    }
    return ("candidate" if all(conds.values()) else "rejected"), conds


# --------------------------------------------------------------------------- #
# Build + the 6 pinned looks on the cached data
# --------------------------------------------------------------------------- #

def main() -> None:
    c16_events = pd.read_parquet(CACHE_DIR / C16_EVENTS_CACHE)
    market = pd.read_parquet(CACHE_DIR / "market_data.parquet",
                             columns=["date", "ticker", "close", "volume"])
    factors = load_factors()

    series, qual, stats = build_c20(c16_events, market)
    series.to_parquet(CACHE_DIR / SERIES_CACHE)
    qual.to_parquet(CACHE_DIR / EVENTS_CACHE, index=False)

    n_tr = int(qual["tradeable"].sum())
    logger.info(f"qualifying events: {len(qual)} of {len(c16_events)} "
                f"({n_tr} tradeable of "
                f"{int(c16_events['tradeable'].sum())}), "
                f"{qual['ticker'].nunique()} tickers; "
                f"conc_count median {qual['conc_count'].median():.0f} "
                f"(full-list median {stats['conc_count'].median():.0f}); "
                f"warmup-blocked {int((stats['n_prior_3y'] < MIN_PRIOR_3Y).sum())}")
    qt = qual.loc[qual["tradeable"]]
    episodes = (qt["event_date"].dt.to_period("Q").value_counts()
                .sort_values(ascending=False))
    logger.info("top episodes (tradeable, by quarter):\n"
                + episodes.head(10).to_string())
    active = series["n_active"] >= 1
    logger.info(f"series: {len(series)}d, days with >=1 name "
                f"{active.mean():.1%}, avg names when active "
                f"{series.loc[active, 'n_active'].mean():.1f}, "
                f"max {int(series['n_active'].max())}")

    # Looks 1-3: FF5+MOM full sample + midpoint-date halves (C16 machinery).
    looks = pinned_looks(series, factors=factors)
    for k, res in enumerate(looks, start=1):
        logger.info(
            f"C20 look {k}/6 ({res['window']} {res['start']}->{res['end']}, "
            f"{res['n_days']}d): alpha {res['alpha_ann']:+.2%}/yr "
            f"(HAC t {res['alpha_t_hac']:+.2f}), "
            f"mkt beta {res['loadings']['mkt_rf']:+.3f}, R2 {res['r2']:.3f}, "
            f"avg active {res['avg_active']:.2f}, "
            f"days with >=1 name {res['pct_days_active']:.1%}")

    # Look 4: within-ticker date-permutation placebo, median of seeds {0,1,2}.
    perm = permutation_look(qual, market, factors=factors)
    per_seed = ", ".join(
        f"seed {r['seed']}: {r['alpha_ann']:+.2%}/yr (t {r['alpha_t_hac']:+.2f}, "
        f"n {r['n_events']})" for r in perm["per_seed"])
    logger.info(f"C20 look 4/6 (within-ticker permutation placebo): "
                f"{per_seed}; median alpha {perm['alpha_ann_median']:+.2%}/yr "
                f"(median t {perm['alpha_t_hac_median']:+.2f}); real - median "
                f"= {looks[0]['alpha_ann'] - perm['alpha_ann_median']:+.2%}/yr")

    # Look 5: same-ticker -252td matched control, paired daily diff.
    m = matched_control_look(qual, c16_events, market)
    logger.info(f"C20 look 5/6 (same-ticker -{CTRL_SHIFT_TD}td matched "
                f"control): matched {m['n_matched']}/{m['n_events']} "
                f"= {m['match_rate']:.1%} (skips: {m['skips']['edge']} edge, "
                f"{m['skips']['px']} no price, {m['skips']['collide']} "
                f"collide); paired diff real-control "
                f"{m['diff_mean_ann']:+.2%}/yr, HAC t {m['diff_t_hac']:+.2f} "
                f"(lags {PAIRED_HAC_LAGS}, {m['n_days']}d)")

    # Look 6: same-window non-event control (decisive).
    ne = nonevent_control_look(qual, c16_events, market, factors=factors)
    logger.info(f"C20 look 6/6 (same-window non-event control, seed "
                f"{NONEVENT_SEED}): matched {ne['n_matched']}/{ne['n_events']} "
                f"({ne['n_unmatched']} unmatched, {ne['n_ctrl_names']} "
                f"distinct control names); control book alpha "
                f"{ne['ctrl_alpha_ann']:+.2%}/yr (HAC t "
                f"{ne['ctrl_alpha_t_hac']:+.2f}), mkt beta "
                f"{ne['ctrl_mkt_beta']:+.3f}, R2 {ne['ctrl_r2']:.3f}; "
                f"paired diff real-nonevent {ne['diff_mean_ann']:+.2%}/yr, "
                f"HAC t {ne['diff_t_hac']:+.2f} (lags {PAIRED_HAC_LAGS}, "
                f"{ne['n_days']}d) — the decisive number")

    status, conds = decide(looks[0], perm, ne)
    logger.info(
        f"pinned bar: full t >= +{FULL_T_BAR} "
        f"[{'PASS' if conds['full_t'] else 'FAIL'}], real - perm-median >= "
        f"+{PERM_GAP_BAR:.0%}/yr [{'PASS' if conds['perm_gap'] else 'FAIL'}], "
        f"paired t >= +{PAIRED_T_BAR} "
        f"[{'PASS' if conds['paired_t'] else 'FAIL'}] -> {status}")


if __name__ == "__main__":
    main()
