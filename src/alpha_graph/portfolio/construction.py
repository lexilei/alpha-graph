"""Cross-sectional portfolio construction: quantile long-short weights.

Conventions
-----------
- Weights are fractions of portfolio NAV. The long leg sums to +1.0 and the
  short leg to -1.0: dollar-neutral, gross exposure 2.0.
- Quantile bins are assigned by stable rank on (score, ticker): ties are
  broken by ticker string, so identical inputs always produce identical
  portfolios (no hash/order nondeterminism).
- NaN scores are excluded before ranking.
- Within each leg names start equal-weighted; `max_weight` then caps each
  name with the excess redistributed pro-rata among uncapped names
  (`cap_and_redistribute`). With equal starting weights the cap either does
  not bind or binds for every name at once, so redistribution is a no-op
  there; the redistribution path is generic so non-equal schemes inherit it.
  An infeasible cap (leg_size * max_weight < 1) raises instead of silently
  shrinking gross exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-15


def cap_and_redistribute(weights: pd.Series, cap: float) -> pd.Series:
    """Cap non-negative weights at `cap`, redistributing excess pro-rata.

    Preserves the input total (water-filling: violators are fixed at the cap,
    the remaining mass is re-spread proportionally over uncapped names,
    repeated until no violation). Raises ValueError if the cap is infeasible
    (len(weights) * cap < weights.sum()).
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    w = weights.astype(float).copy()
    if (w < 0).any():
        raise ValueError("cap_and_redistribute expects non-negative weights")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must have positive total")
    if len(w) * cap < total * (1.0 - 1e-12):
        raise ValueError(
            f"cap {cap} infeasible: {len(w)} names cannot hold total {total:.6g}"
        )
    fixed = pd.Series(False, index=w.index)
    for _ in range(len(w) + 1):
        over = (w > cap + _EPS) & ~fixed
        if not over.any():
            break
        fixed |= over
        w[fixed] = cap
        residual = total - cap * float(fixed.sum())
        free = ~fixed
        if free.any():
            free_sum = float(w[free].sum())
            if free_sum > 0:
                w[free] *= residual / free_sum
            else:  # degenerate: spread residual equally over free names
                w[free] = residual / float(free.sum())
    if (w > cap * (1.0 + 1e-9)).any():
        raise RuntimeError("cap redistribution failed to converge")
    return w


def quantile_ls_weights(
    scores: pd.Series,
    n_quantiles: int = 10,
    direction: int = 1,
    max_weight: float = 0.05,
) -> pd.Series:
    """Long-short weights from one cross-section of scores.

    Ranks `scores` (index = ticker), longs the top quantile and shorts the
    bottom quantile, equal weight within each leg, each leg summing to 1.0
    gross (dollar-neutral, gross 2.0). `direction=-1` flips the sign of every
    weight (for signals where LOW predicts high returns). Per-name cap
    `max_weight` is enforced with redistribution (see cap_and_redistribute).

    NaN scores are excluded; fewer than `n_quantiles` valid names raises.
    Ties are handled deterministically by stable rank on (score, ticker).
    """
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {direction}")
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    if not 0 < max_weight <= 1:
        raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
    s = scores.dropna().astype(float)
    n = len(s)
    if n < n_quantiles:
        raise ValueError(f"{n} non-NaN scores < n_quantiles={n_quantiles}")
    frame = pd.DataFrame({"score": s.to_numpy(), "name": [str(t) for t in s.index]}, index=s.index)
    order = frame.sort_values(["score", "name"], kind="mergesort").index
    bins = (np.arange(n) * n_quantiles) // n  # exact integer binning, rank 0 = lowest score
    short_names = order[bins == 0]
    long_names = order[bins == n_quantiles - 1]
    w_long = cap_and_redistribute(
        pd.Series(1.0 / len(long_names), index=long_names), max_weight
    )
    w_short = cap_and_redistribute(
        pd.Series(1.0 / len(short_names), index=short_names), max_weight
    )
    w = pd.concat([w_long, -w_short]).sort_index() * float(direction)
    w.name = "weight"
    return w
