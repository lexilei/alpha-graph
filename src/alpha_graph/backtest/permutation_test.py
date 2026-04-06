"""Permutation test for the Anti-Momentum Features extension.

The big question after the multi-testing correction landed: is the
Anti-Momentum Sharpe of ~1.95 a real signal or 23-month luck?

This module answers that by running a permutation test:
  1. Compute the real `dist_52w_high` and `reversal_5d` features.
  2. For each of N permutations, jointly shuffle those two feature columns
     within each month (preserving the marginal distribution and within-row
     correlation, but breaking the link to ticker and forward return).
  3. Retrain the same walk-forward LightGBM on the shuffled features.
  4. Run the same long/short backtest and record the Sharpe.
  5. Compute the percentile of the real Sharpe in the resulting null
     distribution.

Interpretation:
  - Real Sharpe in the >95th percentile of the null → features carry real
    signal that the multi-testing correction was just unable to detect at
    23 months. Justifies expanding the sample.
  - Real Sharpe in the 50-95th percentile → ambiguous, more data needed.
  - Real Sharpe below 50th percentile → the features are pure overfit and
    the 1.95 number should be retracted.

Usage:
    python -m alpha_graph.backtest.permutation_test --n 200
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.backtest.extensions import (
    LGBM_EXT_PARAMS,
    _load_data,
    _metrics,
    _run_ls,
)
from alpha_graph.config import CACHE_DIR, set_global_seeds


FEATURE_COLS = ["dist_52w_high", "reversal_5d"]
MODEL_FEATURES = ["signal", "dist_52w_high", "reversal_5d"]


def build_enhanced_panel(preds: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Compute the real anti-momentum features and merge into predictions.

    Mirrors ext1_anti_momentum_features lines 124-143 exactly so the
    permutation test runs against the same data the headline number used.
    """
    market = market.sort_values(["ticker", "date"]).copy()

    # 52-week high distance: (price / 252-day max) - 1
    market["high_252d"] = market.groupby("ticker")["close"].transform(
        lambda s: s.rolling(252, min_periods=63).max()
    )
    market["dist_52w_high"] = market["close"] / market["high_252d"] - 1

    # Short-term reversal (negative of trailing 5-day return)
    market["reversal_5d"] = -market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(5)
    )

    market["month"] = market["date"].dt.to_period("M")
    new_feat = market.groupby(["ticker", "month"]).last()[FEATURE_COLS].reset_index()

    return preds.merge(new_feat, on=["ticker", "month"], how="left")


def shuffle_features_within_month(
    enhanced: pd.DataFrame,
    feature_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Joint within-month permutation of feature columns.

    For each month, the same row-permutation is applied to *all* feature
    columns simultaneously. This preserves:
      - the marginal distribution of every feature
      - the within-row joint distribution between features (e.g. high
        dist_52w_high tends to come with high reversal_5d)
    But it breaks:
      - the link between (feature value) and (ticker, forward return)
      - any cross-sectional rank information

    This is the right null for "do these features have predictive content?"
    """
    out = enhanced.copy()
    for month in out["month"].unique():
        idx = out.index[out["month"] == month].tolist()
        if len(idx) <= 1:
            continue
        perm = rng.permutation(len(idx))
        for col in feature_cols:
            vals = out.loc[idx, col].values
            out.loc[idx, col] = vals[perm]
    return out


def retrain_and_score(enhanced: pd.DataFrame) -> float:
    """Walk-forward retrain LightGBM and return the L/S Sharpe.

    Mirrors ext1_anti_momentum_features lines 152-186 — same training
    window, same hyperparameters, same evaluation. The only thing that
    can change between calls is the *content* of the feature columns
    (real vs shuffled).
    """
    import lightgbm as lgb

    months = sorted(enhanced["month"].unique())
    train_results = []

    for i, test_month in enumerate(months):
        train_months = months[max(0, i - 12):i]
        if len(train_months) < 6:
            continue

        train = enhanced[enhanced["month"].isin(train_months)].dropna(
            subset=MODEL_FEATURES + ["actual_return"]
        )
        test = enhanced[enhanced["month"] == test_month].dropna(
            subset=MODEL_FEATURES + ["actual_return"]
        )
        if len(train) < 100 or len(test) < 10:
            continue

        X_tr = train[MODEL_FEATURES].values
        y_tr = train["actual_return"].values
        X_te = test[MODEL_FEATURES].values

        model = lgb.LGBMRegressor(**LGBM_EXT_PARAMS)
        model.fit(X_tr, y_tr)
        test = test.copy()
        test["enhanced_signal"] = model.predict(X_te)
        train_results.append(test)

    if not train_results:
        return float("nan")

    enhanced_preds = pd.concat(train_results, ignore_index=True)
    enhanced_preds["signal"] = enhanced_preds["enhanced_signal"]
    df = _run_ls(enhanced_preds)
    return float(_metrics(df)["sharpe"])


def run_permutation_test(n_permutations: int = 200) -> dict:
    """Run the full permutation test and return results.

    Returns a dict with:
      - real_sharpe: the unshuffled Sharpe (should match the headline)
      - null_sharpes: array of N shuffled Sharpes
      - percentile: where real falls in the null distribution (0-100)
      - p_value: one-sided p-value (fraction of null >= real)
      - null_mean, null_std: summary stats of the null distribution
    """
    set_global_seeds()
    preds, market, _regimes = _load_data()
    enhanced = build_enhanced_panel(preds, market)

    logger.info("Running real (unshuffled) backtest...")
    real_sharpe = retrain_and_score(enhanced)
    logger.info(f"Real Sharpe = {real_sharpe:.4f}")

    null_sharpes = []
    rng_master = np.random.default_rng(42)
    for k in range(n_permutations):
        # Each permutation gets its own RNG seeded from the master sequence,
        # so the test as a whole is reproducible but the perms are independent.
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        shuffled = shuffle_features_within_month(enhanced, FEATURE_COLS, rng)
        s = retrain_and_score(shuffled)
        null_sharpes.append(s)
        if (k + 1) % 25 == 0:
            arr = np.array([x for x in null_sharpes if not np.isnan(x)])
            logger.info(
                f"  {k + 1}/{n_permutations}  null mean={arr.mean():+.3f}  "
                f"null std={arr.std():.3f}  real percentile so far="
                f"{(arr < real_sharpe).mean() * 100:.1f}"
            )

    null_arr = np.array([x for x in null_sharpes if not np.isnan(x)])
    percentile = float((null_arr < real_sharpe).mean() * 100)
    p_value = float((null_arr >= real_sharpe).mean())

    return {
        "real_sharpe": real_sharpe,
        "null_sharpes": null_arr,
        "percentile": percentile,
        "p_value": p_value,
        "null_mean": float(null_arr.mean()),
        "null_std": float(null_arr.std()),
        "null_min": float(null_arr.min()),
        "null_max": float(null_arr.max()),
        "n_valid": int(len(null_arr)),
    }


def interpret(result: dict) -> str:
    """Verdict text based on the percentile of the real Sharpe."""
    pct = result["percentile"]
    if pct >= 99:
        return (
            "STRONG real signal. The headline 1.95 is in the top 1% of the "
            "null. The MTC failure (Bonferroni p=0.065) is purely a sample-size "
            "issue, not a fake signal. Highest priority: expand to SP500 + "
            "backfill Lazy Prices to 2010."
        )
    if pct >= 95:
        return (
            "LIKELY real signal. Real Sharpe in top 5% of null. "
            "Expanding the sample is the right next step."
        )
    if pct >= 75:
        return (
            "WEAK evidence. Real Sharpe is above the median but well within "
            "the bulk of the null. The 1.95 number is not robust — keep the "
            "feature only with caveats and prioritize a 50/50 in/out validation."
        )
    if pct >= 50:
        return (
            "AMBIGUOUS. The features add no detectable signal beyond what "
            "random shuffling produces. Consider retracting Anti-Momentum "
            "from the headline narrative."
        )
    return (
        "OVERFIT. Real Sharpe is BELOW the null median, meaning random "
        "features actually do better on average. Drop Anti-Momentum entirely."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200,
                        help="Number of permutations (default 200)")
    args = parser.parse_args()

    result = run_permutation_test(n_permutations=args.n)

    print()
    print("=" * 72)
    print("  PERMUTATION TEST — Anti-Momentum Features")
    print("=" * 72)
    print(f"  Permutations:    {args.n}")
    print(f"  Real Sharpe:     {result['real_sharpe']:+.4f}")
    print(f"  Null mean:       {result['null_mean']:+.4f}")
    print(f"  Null std:        {result['null_std']:.4f}")
    print(f"  Null range:      [{result['null_min']:+.3f}, {result['null_max']:+.3f}]")
    print(f"  Real percentile: {result['percentile']:.1f}%")
    print(f"  One-sided p:     {result['p_value']:.4f}")
    print("-" * 72)
    print(f"  Verdict: {interpret(result)}")
    print("=" * 72)

    # Persist null distribution for plotting / reuse
    out_path = CACHE_DIR / "permutation_test_null.parquet"
    pd.DataFrame({"null_sharpe": result["null_sharpes"]}).to_parquet(
        out_path, index=False
    )
    logger.info(f"Saved null distribution to {out_path}")


if __name__ == "__main__":
    main()
