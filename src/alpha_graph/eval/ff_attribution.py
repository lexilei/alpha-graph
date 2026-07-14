"""Factor-model attribution: regress a strategy's daily P&L on the Fama-French
5 factors + momentum (or CAPM), with HAC/Newey-West coefficient errors.

Why not `ic_tools.hac_tstat`: that helper computes a Newey-West t for the MEAN
of a single series. Attribution needs a HAC covariance for REGRESSION
coefficients, so this module uses statsmodels OLS with cov_type="HAC"
(same Bartlett kernel, applied to the score series x_t * e_t).

Factor data: data/cache/ff5_mom_daily.parquet, built by
scripts/fetch_ff_factors.py — decimal daily returns with columns
mkt_rf/smb/hml/rmw/cma/mom/rf, inner-joined FF5 (2x3) + momentum from the
Ken French library. Pass a `factors` frame directly to stay offline (tests do).

Conventions
-----------
- `pnl` is a daily TOTAL return series by default; the regression is run on
  pnl - rf. Pass excess=True when pnl is already an excess or self-financing
  series (e.g. the net of a dollar-neutral long-short book): rf subtraction
  is then skipped, because subtracting it again misstates alpha by ~-rf.
  A factors frame without an `rf` column gets rf=0 with a warning
  (excess=False only; excess=True never uses rf).
- Inputs are DECIMAL daily returns. If the aligned pnl or mkt_rf series has
  a daily std above 0.2 (20%/day — no real strategy or factor), a ValueError
  is raised: that scale means percent-scaled data leaked in, which would
  otherwise scale every alpha and beta by 100 silently.
- Alignment is an inner join on normalized dates; `n_days` is the sample the
  OLS actually saw and `n_dropped` counts non-NaN pnl days that fell out
  (outside factor coverage or factor NaN). A loss above 10% logs a warning.
- Default HAC lags = ceil(1.5 * 21) = 32 trading days: 1.5 months, covering
  the residual autocorrelation of a monthly-rebalanced book with margin.
  Override via `lags` for faster/slower books.
- Annualization is arithmetic: alpha_ann = alpha * 252,
  resid_vol_ann = std(resid, ddof=1) * sqrt(252).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[3]
FF5_MOM_PARQUET = _REPO_ROOT / "data" / "cache" / "ff5_mom_daily.parquet"

MODELS = {
    "ff5_mom": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"],
    "capm": ["mkt_rf"],
}

DEFAULT_HAC_LAGS = int(math.ceil(1.5 * 21))  # 32


def load_factors(path: str | Path = FF5_MOM_PARQUET) -> pd.DataFrame:
    """Load the cached daily factor file (decimal returns, DatetimeIndex)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/fetch_ff_factors.py first")
    f = pd.read_parquet(path)
    f.index = pd.DatetimeIndex(f.index)
    return f


def _normalize_index(idx: pd.Index, name: str) -> pd.DatetimeIndex:
    """tz-naive midnight stamps so equal calendar days join; refuse duplicates."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    if idx.has_duplicates:
        raise ValueError(f"{name} index has duplicate dates")
    return idx


def ff_attribution(pnl: pd.Series, factors: pd.DataFrame | None = None,
                   model: str = "ff5_mom", lags: int | None = None,
                   excess: bool = False) -> dict:
    """OLS of daily excess P&L on a factor model, HAC (Newey-West) t-stats.

    Parameters
    ----------
    pnl : daily strategy returns (decimal, DatetimeIndex); total returns by
        default, an already-excess series with excess=True.
    factors : daily factor frame with the model's columns (+ optionally `rf`),
        decimal returns. None loads data/cache/ff5_mom_daily.parquet.
    model : "ff5_mom" (mkt_rf/smb/hml/rmw/cma/mom) or "capm" (mkt_rf only).
    lags : HAC maxlags; None -> DEFAULT_HAC_LAGS = ceil(1.5*21) = 32.
    excess : False (default) treats pnl as TOTAL returns and regresses
        pnl - rf — use for long-only books, whose returns include the
        risk-free component. True treats pnl as ALREADY excess and skips the
        rf subtraction — use for self-financing dollar-neutral long-short
        spreads, where subtracting rf again would understate alpha by ~rf.

    Returns
    -------
    dict with: model, n_days, n_pnl_days, n_dropped, hac_lags,
    alpha (daily), alpha_ann (x252), alpha_se_hac, alpha_t_hac,
    loadings {factor: beta}, loadings_t_hac {factor: t}, r2,
    resid_vol_ann, ir_resid (alpha_ann / resid_vol_ann).
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:  # pragma: no cover
        raise ImportError("ff_attribution needs statsmodels "
                          "(dev extra: pip install statsmodels)") from e

    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    if not isinstance(pnl, pd.Series):
        raise TypeError("pnl must be a pd.Series of daily returns")
    cols = MODELS[model]

    if factors is None:
        factors = load_factors()
    missing = [c for c in cols if c not in factors.columns]
    if missing:
        raise ValueError(f"factors frame is missing columns {missing}")

    pnl = pnl.copy()
    pnl.index = _normalize_index(pnl.index, "pnl")
    f = factors.copy()
    f.index = _normalize_index(f.index, "factors")

    if "rf" in f.columns:
        rf = f["rf"]
    else:
        if not excess:
            logger.warning("factors frame has no 'rf' column — regressing pnl "
                           "un-differenced (rf assumed 0)")
        rf = pd.Series(0.0, index=f.index)

    joined = pd.concat([pnl.rename("pnl"), f[cols], rf.rename("rf")],
                       axis=1, join="inner").dropna().sort_index()
    n_days = len(joined)
    n_pnl = int(pnl.notna().sum())
    n_dropped = n_pnl - n_days
    if n_days < len(cols) + 2:
        raise ValueError(f"only {n_days} overlapping days after the inner "
                         f"join — too few to regress {len(cols)} factors")
    if n_dropped > 0.10 * n_pnl:
        logger.warning(f"alignment dropped {n_dropped}/{n_pnl} pnl days "
                       f"({n_dropped / n_pnl:.0%}) — check factor coverage "
                       f"(French library lags ~2 months)")

    # Unit guard: decimal daily returns have std far below 20%/day; above 0.2
    # the input is percent-scaled (x100), which would silently scale every
    # alpha and beta by 100 (a planted beta of 0.5 reports as ~50).
    for label, series in (("pnl", joined["pnl"]),
                          ("factors' mkt_rf", joined["mkt_rf"])):
        sd = float(series.std())
        if sd > 0.2:
            raise ValueError(
                f"{label} daily std {sd:.3g} exceeds 0.2 (20%/day) — no real "
                f"return series does this; the input is likely percent-scaled, "
                f"pass decimal returns (divide by 100)")

    hac_lags = DEFAULT_HAC_LAGS if lags is None else int(lags)
    # rf stays in the join in both modes so the regression sample is identical;
    # excess=True only skips the subtraction.
    y = joined["pnl"] if excess else joined["pnl"] - joined["rf"]
    X = sm.add_constant(joined[cols])
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    alpha = float(res.params["const"])
    alpha_ann = alpha * 252.0
    resid_vol_ann = float(np.std(res.resid, ddof=1) * math.sqrt(252.0))
    return {
        "model": model,
        "n_days": int(n_days),
        "n_pnl_days": n_pnl,
        "n_dropped": int(n_dropped),
        "hac_lags": int(hac_lags),
        "alpha": alpha,
        "alpha_ann": alpha_ann,
        "alpha_se_hac": float(res.bse["const"]),
        "alpha_t_hac": float(res.tvalues["const"]),
        "loadings": {c: float(res.params[c]) for c in cols},
        "loadings_t_hac": {c: float(res.tvalues[c]) for c in cols},
        "r2": float(res.rsquared),
        "resid_vol_ann": resid_vol_ann,
        "ir_resid": alpha_ann / resid_vol_ann if resid_vol_ann > 0 else float("nan"),
    }
