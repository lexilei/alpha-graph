"""Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) for Method E.

The "Probabilistic Sharpe Ratio" PSR(SR*) is the probability that the true
Sharpe exceeds a benchmark SR*, given the observed Sharpe, sample size, and
the third/fourth moments of the return distribution. The "Deflated Sharpe
Ratio" DSR adjusts for multiple-testing: when N strategies are tried, the
expected maximum Sharpe under the null is non-zero, so the benchmark SR* is
set to that expected maximum instead of 0.

In this project we tried 11 method variants (baseline + A-J) and selected
Method E. DSR tells us the probability that Method E's Sharpe is real after
accounting for that selection.

Reference: Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-
Normality. Journal of Portfolio Management, 40(5), 94-107.

Usage:
    python -m alpha_graph.backtest.deflated_sharpe
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from alpha_graph.backtest.attribution import reproduce_method_e
from alpha_graph.backtest.extensions import _load_data
from alpha_graph.backtest.pit_universe import fetch_membership, filter_preds_pit
from alpha_graph.config import CACHE_DIR, set_global_seeds

EULER_MASCHERONI = 0.5772156649015329


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """Plain annualized Sharpe (no excess return adjustment — assumes returns are already net)."""
    mu = returns.mean()
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return (mu / sd) * np.sqrt(periods_per_year)


def expected_max_sharpe(n_trials: int, std_sharpe_across_trials: float) -> float:
    """Bailey-Lopez de Prado eqn for E[max SR] over N i.i.d. trials with mean 0
    and standard deviation std_sharpe_across_trials.
    """
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return std_sharpe_across_trials * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def probabilistic_sharpe(
    sr_obs: float,
    sr_benchmark: float,
    n_periods: int,
    skew: float,
    kurt_excess: float,
) -> float:
    """PSR(SR*) — probability that true SR > SR*, given observed SR_obs and
    higher moments. Bailey-Lopez de Prado equation.

    Note: kurt_excess is excess kurtosis (= kurt - 3 for a normal distribution).
    """
    if n_periods <= 1:
        return float("nan")
    numerator = (sr_obs - sr_benchmark) * math.sqrt(n_periods - 1)
    denominator = math.sqrt(
        1.0 - skew * sr_obs + (kurt_excess / 4.0) * sr_obs ** 2
    )
    return float(stats.norm.cdf(numerator / denominator))


def compute_dsr(
    returns: np.ndarray,
    n_trials: int,
    trial_sharpes: np.ndarray | None = None,
    periods_per_year: int = 12,
) -> dict:
    """Compute SR, PSR(0), and DSR for a return stream.

    `trial_sharpes` is the array of (annualized) Sharpe ratios for ALL trials
    that were considered before selection (e.g. the 11 method variants). If
    None, we fall back to assuming std_sharpe_across_trials = 1, which is the
    most conservative case — strategies are assumed to vary by 1 unit of
    Sharpe under the null.
    """
    n = len(returns)
    sr_ann = annualized_sharpe(returns, periods_per_year)
    sr_per_period = sr_ann / np.sqrt(periods_per_year)  # the formula uses non-annualized SR

    skew = float(stats.skew(returns, bias=False))
    kurt_excess = float(stats.kurtosis(returns, fisher=True, bias=False))

    psr_zero_per_period = probabilistic_sharpe(sr_per_period, 0.0, n, skew, kurt_excess)

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        std_trials = float(np.std(trial_sharpes / np.sqrt(periods_per_year), ddof=1))
        emax = expected_max_sharpe(n_trials, std_trials)
    else:
        # Conservative default: assume trial Sharpes have unit std (per period scale)
        std_trials = 1.0 / np.sqrt(periods_per_year)
        emax = expected_max_sharpe(n_trials, std_trials)

    dsr_per_period = probabilistic_sharpe(sr_per_period, emax, n, skew, kurt_excess)

    return {
        "n_periods": n,
        "sr_annualized": sr_ann,
        "sr_per_period": sr_per_period,
        "skew": skew,
        "kurt_excess": kurt_excess,
        "n_trials": n_trials,
        "trial_sharpes_std_per_period": std_trials,
        "expected_max_sharpe_per_period": emax,
        "expected_max_sharpe_annualized": emax * np.sqrt(periods_per_year),
        "psr_above_zero": psr_zero_per_period,
        "dsr_above_max": dsr_per_period,
    }


def _print_block(label: str, info: dict) -> None:
    print()
    print("=" * 80)
    print(f"  {label}")
    print("=" * 80)
    print(f"  Sample length:                   {info['n_periods']} months")
    print(f"  Annualized Sharpe (observed):    {info['sr_annualized']:+.3f}")
    print(f"  Skewness of returns:             {info['skew']:+.3f}")
    print(f"  Excess kurtosis of returns:      {info['kurt_excess']:+.3f}")
    print()
    print(f"  N trials considered:             {info['n_trials']}")
    print(f"  Std of trial Sharpes (per-period): {info['trial_sharpes_std_per_period']:.4f}")
    print(f"  E[max SR] over N trials (ann):   {info['expected_max_sharpe_annualized']:+.3f}")
    print()
    print(f"  PSR(SR > 0):                     {info['psr_above_zero']*100:.2f}%")
    print(f"  DSR (SR > E[max SR | N trials]): {info['dsr_above_max']*100:.2f}%")
    print()
    if info["dsr_above_max"] > 0.95:
        verdict = "STRONG: DSR > 95%, alpha highly unlikely to be a multiple-testing artifact"
    elif info["dsr_above_max"] > 0.90:
        verdict = "MODERATE: DSR 90-95%, alpha probably real but caveat selection"
    elif info["dsr_above_max"] > 0.50:
        verdict = "WEAK: DSR 50-90%, signal might be lucky from trying N variants"
    else:
        verdict = "FAIL: DSR < 50%, indistinguishable from selection bias"
    print(f"  VERDICT: {verdict}")
    print("=" * 80)


def _method_e_returns(preds: pd.DataFrame, cost: float = 0.001) -> np.ndarray:
    months = sorted(preds["month"].unique())
    rows = []
    for month in months:
        m = preds[preds["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue
        sorted_m = m.sort_values("signal", ascending=False)
        long_r = float(sorted_m.head(10)["actual_return"].mean())
        rows.append(long_r - cost)
    return np.array(rows)


def main():
    set_global_seeds()
    preds, _market, _regimes = _load_data()

    # Trial Sharpes from the 11 variants in improvements.py (we tried these
    # before selecting Method E). Loaded from cached results.
    trials_path = CACHE_DIR / "improvements_results.parquet"
    trial_sharpes = None
    if trials_path.exists():
        trials = pd.read_parquet(trials_path)
        trial_sharpes = trials["sharpe"].dropna().values
        logger.info(
            f"Loaded {len(trial_sharpes)} trial Sharpes from improvements_results: "
            f"min={trial_sharpes.min():.2f}, max={trial_sharpes.max():.2f}, "
            f"std={trial_sharpes.std(ddof=1):.2f}"
        )

    n_trials = len(trial_sharpes) if trial_sharpes is not None else 11

    # ----- Full universe Method E -----
    rets_full = _method_e_returns(preds)
    info_full = compute_dsr(rets_full, n_trials=n_trials, trial_sharpes=trial_sharpes)
    _print_block("METHOD E — FULL UNIVERSE", info_full)

    # ----- PIT universe Method E -----
    membership = fetch_membership()
    pit_preds = filter_preds_pit(preds, membership)
    rets_pit = _method_e_returns(pit_preds)
    info_pit = compute_dsr(rets_pit, n_trials=n_trials, trial_sharpes=trial_sharpes)
    _print_block("METHOD E — PIT UNIVERSE (the honest read-out)", info_pit)

    # Save
    df = pd.DataFrame([
        {"variant": "full_universe", **info_full},
        {"variant": "pit_universe", **info_pit},
    ])
    df.to_parquet(CACHE_DIR / "method_e_dsr.parquet", index=False)
    logger.info("Saved DSR results")


if __name__ == "__main__":
    main()
