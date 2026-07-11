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
    return int(min(base, max_lags))


def emax_null(n_trials: int) -> float:
    """Expected maximum of n_trials iid N(0,1) draws under the null
    (Bailey-Lopez de Prado two-term approximation). n_trials<=1 -> 0.
    Feed this the LEDGER look-count, not effective_n's m_eff."""
    if n_trials <= 1:
        return 0.0
    return float((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                 + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * math.e)))


def deflated_t(t: float, n_trials: int, n_obs: int | None = None) -> dict:
    """|t| measured against the best-of-N null ceiling.

    Returns emax_null, the margin |t| - emax_null, and Phi(margin) — the
    probability that the observed t exceeds the expected best of n_trials
    null draws. Approximations (documented, conservative when n_trials is
    the full ledger count): the null t is ~N(0,1); trials independent.
    n_obs is accepted for signature stability (small-sample refinements later).
    """
    e = emax_null(n_trials)
    margin = abs(float(t)) - e
    return {"emax_null": e, "margin": margin,
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
    lags = hac_lags if hac_lags is not None else default_lags(
        horizon_days, sampling_interval_days, ic=x)
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
    for mo, g in panel_m.groupby("month"):
        sub = g[[factor, target, "ticker"]].dropna(subset=[factor, target])
        if len(sub) < min_names:
            continue
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
        "break_even_cost_bps": 1e4 * mean_ls / denom if denom > 0 else float("inf"),
    }


# --------------------------------------------------------------------------- #
# Factor-set diagnostics
# --------------------------------------------------------------------------- #

def effective_n(panel_m: pd.DataFrame, factors: list[str],
                min_names: int = 30) -> dict:
    """Effective number of independent factor COLUMNS (Li-Ji and Nyholt) from
    the pooled cross-sectional rank correlations.

    This measures redundancy among the given factors. It is NOT the
    multiple-testing look-count: the significance ceiling must use the ledger
    N via emax_null(ledger_N) (review S7).
    """
    pooled = []
    for _, g in panel_m.groupby("month"):
        sub = g[factors].dropna(how="all")
        if len(sub) < min_names:
            continue
        pooled.append(sub.rank().apply(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * np.nan))
    R = pd.concat(pooled).corr(min_periods=100).fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(R, 1.0)
    ev = np.linalg.eigvalsh(R)
    ev = ev[ev > 0]
    m = len(factors)
    m_lj = float(sum(1.0 if e >= 1 else e - math.floor(e) for e in ev))
    m_ny = float(1 + (m - 1) * (1 - np.var(ev) / m))
    return {"m_eff_li_ji": m_lj, "m_eff_nyholt": m_ny,
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
