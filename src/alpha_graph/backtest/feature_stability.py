"""Feature importance stability across the 168 walk-forward folds.

The existing ml_combiner reports a single feature importance vector from the
final model. For a robustness claim we need to know: are the same features
important across *all* folds, or does the importance vector drift with
regime?

This script refits the exact same LightGBM walk-forward and records
`feature_importances_` (LightGBM's split-count importance) per fold. Output:
  - `ml_combiner_importance_by_fold.parquet` (long-form: month, feature, split_count)
  - `ml_combiner_importance_heatmap.png`

Usage:
    python -m alpha_graph.backtest.feature_stability
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, set_global_seeds
from alpha_graph.signals.ml_combiner import (
    FEATURE_COLS,
    LGBM_PARAMS,
    PURGE_DAYS,
    TRAIN_MONTHS,
    build_feature_panel,
)


def run_importance_walk_forward() -> pd.DataFrame:
    import lightgbm as lgb

    panel = build_feature_panel()
    panel["month"] = panel["date"].dt.to_period("M")
    months = sorted(panel["month"].unique())
    available = [c for c in FEATURE_COLS if c in panel.columns]
    logger.info(f"Walking forward with features: {available}")

    rows = []
    for i in range(TRAIN_MONTHS, len(months)):
        test_month = months[i]
        train_start = months[max(0, i - TRAIN_MONTHS)]
        train_end = months[i - 1]

        tm = (panel["month"] >= train_start) & (panel["month"] <= train_end)
        train_data = panel[tm]
        if PURGE_DAYS > 0:
            cutoff = train_data["date"].max() - pd.Timedelta(days=PURGE_DAYS)
            train_data = train_data[train_data["date"] <= cutoff]

        test_data = panel[panel["month"] == test_month]
        if len(train_data) < 50 or len(test_data) < 5:
            continue

        X_train = train_data[available]
        y_train = train_data["fwd_return_21d"]
        X_test = test_data[available]
        y_test = test_data["fwd_return_21d"]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        importances = model.feature_importances_
        for feat, imp in zip(available, importances):
            rows.append({
                "month": str(test_month),
                "feature": feat,
                "split_count": int(imp),
            })

    return pd.DataFrame(rows)


def summarize(imp_df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature summary: mean split count, std, fraction of folds selected (>0)."""
    agg = imp_df.groupby("feature")["split_count"].agg(
        mean="mean", std="std", median="median", max="max",
    )
    selected = imp_df.groupby("feature")["split_count"].apply(lambda s: (s > 0).mean())
    agg["fold_selection_rate"] = selected
    return agg.sort_values("mean", ascending=False)


def plot_heatmap(imp_df: pd.DataFrame, out_path) -> None:
    """Heatmap: features (rows) × months (columns), cell = split count."""
    pivot = imp_df.pivot(index="feature", columns="month", values="split_count").fillna(0)
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(16, max(3, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    n_months = pivot.shape[1]
    step = max(1, n_months // 12)
    ax.set_xticks(range(0, n_months, step))
    ax.set_xticklabels([pivot.columns[i] for i in range(0, n_months, step)], rotation=45, ha="right")
    ax.set_xlabel("walk-forward test month")
    ax.set_title("LightGBM split-count feature importance by fold (higher = more splits on that feature)")
    plt.colorbar(im, ax=ax, label="split count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    logger.info(f"Saved heatmap to {out_path}")


def main():
    set_global_seeds()
    imp_df = run_importance_walk_forward()
    if imp_df.empty:
        logger.error("No importance data generated")
        return

    path = CACHE_DIR / "ml_combiner_importance_by_fold.parquet"
    imp_df.to_parquet(path, index=False)
    logger.info(f"Saved per-fold importance ({len(imp_df)} rows) to {path}")

    summary = summarize(imp_df)
    print()
    print("=" * 80)
    print("  FEATURE IMPORTANCE STABILITY ACROSS 156 WALK-FORWARD FOLDS")
    print("=" * 80)
    print(
        f"  {'feature':<20s}  {'mean':>7s}  {'std':>7s}  "
        f"{'median':>7s}  {'max':>5s}  {'fold selected':>14s}"
    )
    print("  " + "-" * 78)
    for feat, row in summary.iterrows():
        print(
            f"  {feat:<20s}  {row['mean']:>7.2f}  {row['std']:>7.2f}  "
            f"{row['median']:>7.1f}  {row['max']:>5.0f}  {row['fold_selection_rate']*100:>12.1f}%"
        )
    print("=" * 80)

    out_fig = CACHE_DIR.parent.parent / "report" / "fig_feature_importance_heatmap.png"
    plot_heatmap(imp_df, out_fig)


if __name__ == "__main__":
    main()
