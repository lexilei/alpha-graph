"""Factor-IC diagnostics: the analyses this project kept hand-rewriting,
as tested functions. See reports/plan_batch1_infra.md (Item 3) for the spec
and the review findings each guardrail answers.

Conventions
-----------
- A "panel_m" is a monthly-sampled panel: one row per (ticker, month), with a
  `month` Period column — the output of `to_monthly` in
  scripts/factor_orthogonality.py.
- All t-statistics on IC series assume the months are the sample unit. When
  the sampling interval is shorter than the holding horizon, adjacent
  observations overlap and a naive t is invalid: functions here refuse to
  emit it and force HAC instead (plan v2 guardrail; the weekly-sampling trap).
- `emax_null` takes the LEDGER look-count N, never `effective_n`'s m_eff:
  m_eff measures redundancy among factor columns, not the number of looks
  taken at the data (review S7).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Cross-sectional transforms (moved verbatim from factor_orthogonality —
# that script imports these; do not alter behavior)
# --------------------------------------------------------------------------- #

def _xs_rank_z(s: pd.Series) -> pd.Series:
    """Cross-sectional rank -> standardized to mean 0 / std 1."""
    r = s.rank()
    if r.notna().sum() < 2 or r.std(ddof=0) == 0:
        return pd.Series(np.nan, index=s.index)
    return (r - r.mean()) / r.std(ddof=0)


def sector_neutralize(sub: pd.DataFrame, cols: list[str],
                      sector_col: str = "sector") -> pd.DataFrame:
    """Demean `cols` within `sector_col` groups (copy; NaN sectors -> own group)."""
    out = sub.copy()
    grp = out[sector_col].fillna("UNKNOWN")
    for c in cols:
        out[c] = out[c] - out.groupby(grp)[c].transform("mean")
    return out


# --------------------------------------------------------------------------- #
# IC series and summaries
# --------------------------------------------------------------------------- #

def monthly_ic(panel_m: pd.DataFrame, factor: str,
               target: str = "fwd_return_21d", min_names: int = 20) -> pd.Series:
    """Per-month cross-sectional Spearman rank-IC between factor and target."""
    out = {}
    for mo, g in panel_m.groupby("month"):
        sub = g[[factor, target]].dropna()
        if len(sub) < min_names:
            continue
        ic = stats.spearmanr(sub[factor], sub[target]).statistic
        if not np.isnan(ic):
            out[mo] = ic
    return pd.Series(out).sort_index()


def hac_tstat(x, lags: int) -> float:
    """Newey-West (Bartlett kernel) t-statistic for the mean of a series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    d = x - mu
    if float(d @ d) == 0.0:
        return float("nan")  # constant series: variance undefined, not +inf
    lrv = float(d @ d) / n
    L = int(min(max(lags, 0), n - 1))
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        lrv += 2.0 * w * float(d[l:] @ d[:-l]) / n
    lrv = max(lrv, 1e-18)
    return float(mu / math.sqrt(lrv / n))


def default_lags(horizon_days: float, sampling_interval_days: float,
                 ic: pd.Series | np.ndarray | None = None,
                 max_lags: int = 24) -> int:
    """HAC lag rule: ceil(horizon / sampling interval), raised by a
    persistence floor — the run of consecutive positive autocorrelations in
    the measured IC series (review S3: the rule must respond to slow factors,
    not only to return overlap)."""
    base = int(math.ceil(horizon_days / float(sampling_interval_days)))
    if ic is not None:
        x = np.asarray(ic, dtype=float)
        x = x[~np.isnan(x)]
        if len(x) > 8:
            d = x - x.mean()
            var = float(d @ d)
            if var > 0:
                run = 0
                for l in range(1, min(max_lags, len(x) // 3) + 1):
                    ac = float(d[l:] @ d[:-l]) / var
                    if ac <= 0:
                        break
                    run = l
                base = max(base, run)
    if base > max_lags:
        logger.warning(f"default_lags: required lags {base} truncated to "
                       f"max_lags={max_lags} — HAC will under-correct; "
                       f"raise max_lags for long-horizon series")
    return int(min(base, max_lags))


def emax_null(n_trials: int) -> float:
    """Expected maximum of n_trials iid N(0,1) draws under the null
    (Bailey-Lopez de Prado two-term approximation). n_trials<=1 -> 0.
    Feed this the LEDGER look-count, not effective_n's m_eff."""
    if n_trials <= 1:
        return 0.0
    return float((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                 + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * math.e)))


def deflated_t(t: float, n_trials: int, n_obs: int | None = None,
               two_sided: bool = True) -> dict:
    """|t| measured against the best-of-N null ceiling.

    two_sided=True (default) compares |t| to E[max of N |Z|] = emax_null(2N):
    selection in this project is over |t| (signs are not pre-registered), and
    the one-sided ceiling understates the null max by ~0.3 sigma at N=25-50
    (review F3, anti-conservative). Set two_sided=False only for a
    pre-registered sign. Null t ~ N(0,1), trials independent (conservative
    when n_trials is the full ledger count). n_obs reserved.
    """
    e = emax_null(2 * n_trials if two_sided else n_trials)
    margin = abs(float(t)) - e
    return {"emax_null": e, "margin": margin, "two_sided": bool(two_sided),
            "prob_exceeds_null_max": float(stats.norm.cdf(margin))}


def ic_summary(ic: pd.Series, horizon_days: float = 21,
               sampling_interval_days: float = 21,
               hac_lags: int | None = None,
               n_trials: int | None = None) -> dict:
    """Summary of a monthly (or weekly) IC series.

    Guardrail: when sampling_interval_days < horizon_days the observations
    overlap and `t_naive` is suppressed (None, naive_suppressed=True) — HAC
    is the only t emitted (plan v2 / review B6).
    """
    x = np.asarray(ic, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return {"n": n, "ok": False}
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    overlap_floor = int(math.ceil(horizon_days / float(sampling_interval_days)))
    if hac_lags is not None:
        # an override may raise but never lower the overlap minimum (review F6)
        lags = max(int(hac_lags), overlap_floor) \
            if sampling_interval_days < horizon_days else int(hac_lags)
    else:
        lags = default_lags(horizon_days, sampling_interval_days, ic=x)
    overlap = sampling_interval_days < horizon_days
    t_naive = None if overlap else (mu / (sd / math.sqrt(n)) if sd > 0 else 0.0)
    out = {
        "ok": True, "n": n, "mean": mu, "std": sd,
        "icir": mu / sd if sd > 0 else 0.0,
        "hit_rate": float((x > 0).mean()),
        "t_naive": t_naive, "naive_suppressed": bool(overlap),
        "hac_lags": lags, "t_hac": hac_tstat(x, lags),
    }
    if n_trials is not None:
        out["deflated"] = deflated_t(out["t_hac"], n_trials, n)
    return out


def split_halves(ic: pd.Series, boundary) -> dict:
    """Split an IC series at `boundary`: first half = index <= boundary,
    second = index > boundary. Semantics pinned by a regression test — an
    off-by-one boundary must change the answer (the 2019-drop bug class)."""
    idx = ic.index
    if isinstance(idx, pd.PeriodIndex):
        boundary = pd.Period(boundary, freq=idx.freq)
    elif isinstance(idx, pd.DatetimeIndex):
        # coerce to the period END so "2019-12" includes Dec-2019 stamps
        # (review F8: naive parse -> Dec 1 -> the 2019-drop bug class)
        boundary = pd.Period(boundary).end_time
    first = ic[idx <= boundary]
    second = ic[idx > boundary]
    def _t(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        if len(x) < 3 or x.std(ddof=1) == 0:
            return float("nan")
        return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
    return {"first_n": len(first), "second_n": len(second),
            "first_mean": float(first.mean()) if len(first) else float("nan"),
            "second_mean": float(second.mean()) if len(second) else float("nan"),
            "first_t": _t(first), "second_t": _t(second)}


# --------------------------------------------------------------------------- #
# Quantile portfolio economics
# --------------------------------------------------------------------------- #

def quantile_ls(panel_m: pd.DataFrame, factor: str,
                target: str = "fwd_return_21d", q: int = 5,
                min_names: int = 50, horizon_days: float = 21,
                sampling_interval_days: float = 21,
                hac_lags: int | None = None) -> dict:
    """Top-minus-bottom quantile long/short economics with turnover and the
    break-even one-way cost.

    Cost convention (stated per review S7): the book is 1 unit long (top
    quantile, equal weight) + 1 unit short (bottom). Replacing fraction f of
    a side trades 2f of that side's notional (sell f, buy f). Monthly drag at
    one-way cost c = 2c(f_top + f_bot). break_even_cost_bps solves drag =
    mean monthly L/S return.
    """
    rows, tops, bots = {}, {}, {}
    tie_months = 0
    for mo, g in panel_m.groupby("month"):
        sub = g[[factor, target, "ticker"]].dropna(subset=[factor, target])
        if len(sub) < min_names:
            continue
        # tie diagnosis: a value mass wider than one quantile straddles a
        # boundary and makes rank(method="first") row-order dependent (F5)
        top_share = sub[factor].value_counts(normalize=True).iloc[0]
        if top_share > 1.0 / q:
            tie_months += 1
        qq = pd.qcut(sub[factor].rank(method="first"), q, labels=False)
        top, bot = sub[qq == q - 1], sub[qq == 0]
        rows[mo] = float(top[target].mean() - bot[target].mean())
        tops[mo], bots[mo] = set(top["ticker"]), set(bot["ticker"])
    ls = pd.Series(rows).sort_index()
    if len(ls) < 3:
        return {"ok": False, "n": len(ls)}

    def _one_way(sets: dict) -> float:
        months = sorted(sets)
        fr = [len(sets[b] - sets[a]) / max(len(sets[b]), 1)
              for a, b in zip(months, months[1:])]
        return float(np.mean(fr)) if fr else float("nan")

    if tie_months:
        logger.warning(f"quantile_ls[{factor}]: tied value mass straddles a "
                       f"quantile boundary in {tie_months}/{len(rows)} months — "
                       f"L/S economics are row-order dependent; treat with care")
    f_top, f_bot = _one_way(tops), _one_way(bots)
    summ = ic_summary(ls, horizon_days, sampling_interval_days, hac_lags)
    mean_ls = float(ls.mean())
    denom = 2.0 * (f_top + f_bot)
    return {
        "ok": True, "n": len(ls), "mean_monthly": mean_ls,
        "t_naive": summ["t_naive"], "naive_suppressed": summ["naive_suppressed"],
        "t_hac": summ["t_hac"], "hac_lags": summ["hac_lags"],
        "hit_rate": float((ls > 0).mean()),
        "ann_sharpe": float(ls.mean() / ls.std(ddof=1) * math.sqrt(12))
        if ls.std(ddof=1) > 0 else float("nan"),
        "turnover_top": f_top, "turnover_bottom": f_bot,
        "tie_months": tie_months,
        "break_even_cost_bps": 1e4 * mean_ls / denom if denom > 0 else float("inf"),
    }


# --------------------------------------------------------------------------- #
# Factor-set diagnostics
# --------------------------------------------------------------------------- #

def ic_decay(panel_m: pd.DataFrame, factor: str, prices: pd.DataFrame,
             horizons: tuple = (5, 10, 21, 42, 63, 126),
             sampling_interval_days: float = 21,
             min_names: int = 20) -> pd.DataFrame:
    """Mean IC and HAC t per holding horizon (plan Item 3; required by Item 4's
    C10 acceptance). `prices` needs columns ticker/date/close; forward returns
    are computed internally per horizon. HAC lags per horizon via default_lags
    with max_lags raised to cover the longest horizon (review F7).
    """
    px = prices[["ticker", "date", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    rows = []
    for h in horizons:
        fwd = px.groupby("ticker")["close"].transform(lambda s: s.shift(-h) / s - 1)
        pxh = px[["ticker", "date"]].assign(fwd=fwd)
        pm = panel_m.merge(pxh, on=["ticker", "date"], how="left")
        ic = monthly_ic(pm, factor, target="fwd", min_names=min_names)
        need = int(math.ceil(h / float(sampling_interval_days)))
        lags = default_lags(h, sampling_interval_days, ic=ic,
                            max_lags=max(24, need))
        rows.append({"horizon_days": h, "n_months": len(ic),
                     "mean_ic": float(ic.mean()) if len(ic) else float("nan"),
                     "hac_lags": lags, "t_hac": hac_tstat(ic, lags)})
    return pd.DataFrame(rows)


def effective_n(panel_m: pd.DataFrame, factors: list[str],
                min_names: int = 30) -> dict:
    """Effective number of independent factor COLUMNS from pooled
    cross-sectional rank correlations. Three estimators (review F4):

    - m_eff_li_ji: the published Li & Ji (2005) f(lam)=I(lam>=1)+(lam-floor(lam))
      — lenient, discounts only near-exact duplicates.
    - m_eff_capped: sum of min(lam, 1) — the stricter variant this project
      used informally (a soft principal-component count).
    - m_eff_nyholt: Nyholt (2004) over ALL M eigenvalues, ddof=1.

    NOT the multiple-testing look-count: the significance ceiling must use
    the ledger N via emax_null. `n_sparse_pairs` counts correlations imputed
    to 0 for insufficient overlap (min_periods) — a nonzero value inflates
    every m_eff; treat the estimates as upper bounds then.
    """
    pooled = []
    for _, g in panel_m.groupby("month"):
        sub = g[factors].dropna(how="all")
        if len(sub) < min_names:
            continue
        pooled.append(sub.rank().apply(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * np.nan))
    C = pd.concat(pooled).corr(min_periods=100)
    n_sparse = int(C.isna().sum().sum())
    R = C.fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(R, 1.0)
    ev = np.linalg.eigvalsh(R)              # full spectrum, ascending
    m = len(factors)
    ev_pos = np.clip(ev, 0.0, None)
    m_li_ji = float(sum((1.0 if e >= 1 else 0.0) + (e - math.floor(e))
                        for e in ev_pos))
    m_capped = float(np.minimum(ev_pos, 1.0).sum())
    m_nyholt = float(1 + (m - 1) * (1 - np.var(ev, ddof=1) / m))
    return {"m_eff_li_ji": m_li_ji, "m_eff_capped": m_capped,
            "m_eff_nyholt": m_nyholt, "n_sparse_pairs": n_sparse,
            "min_eigenvalue": float(ev.min()),
            "eigenvalues": sorted((float(e) for e in ev), reverse=True),
            "note": "column redundancy only; use ledger N for emax_null"}


def rank_autocorr(panel_m: pd.DataFrame, factor: str,
                  warn_threshold: float = 0.65) -> dict:
    """Mean Spearman correlation of consecutive months' cross-sectional
    factor values (aligned on ticker). High persistence => the IC series is
    autocorrelated and a naive t is inflated (the HHI trap); `warn_slow`
    fires at 0.65 (review S3) and the value feeds default_lags' floor."""
    months = sorted(panel_m["month"].unique())
    cors = []
    by_m = {mo: g.set_index("ticker")[factor].dropna()
            for mo, g in panel_m.groupby("month")}
    for a, b in zip(months, months[1:]):
        sa, sb = by_m.get(a), by_m.get(b)
        if sa is None or sb is None:
            continue
        j = pd.concat([sa, sb], axis=1, join="inner", keys=["a", "b"]).dropna()
        if len(j) < 30:
            continue
        c = stats.spearmanr(j["a"], j["b"]).statistic
        if not np.isnan(c):
            cors.append(c)
    mean_ac = float(np.mean(cors)) if cors else float("nan")
    return {"mean_rank_autocorr": mean_ac, "n_pairs": len(cors),
            "warn_slow": bool(mean_ac > warn_threshold) if cors else False}
