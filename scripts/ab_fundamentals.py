"""A/B comparison: ML combiner with vs without fundamental features."""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from alpha_graph.signals.ml_combiner import (
    build_feature_panel, walk_forward_train_predict, FEATURE_COLS,
)
from alpha_graph.backtest.extensions import _load_data, _run_ls, _metrics, ext1_anti_momentum_features

FUND_COLS = [
    "pe_trailing_rank", "pb_ratio_rank", "div_yield_rank",
    "roe_rank", "gross_margin_rank", "debt_equity_rank",
]

def run_variant(label, use_fundamentals):
    """Train ML combiner, then run baseline + anti-momentum extension."""
    import alpha_graph.signals.ml_combiner as mc

    # Temporarily override FEATURE_COLS
    if use_fundamentals:
        mc.FEATURE_COLS = [c for c in FEATURE_COLS]  # all features
    else:
        mc.FEATURE_COLS = [c for c in FEATURE_COLS if c not in FUND_COLS]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Features: {len(mc.FEATURE_COLS)}")
    print(f"{'='*60}")

    # Build panel and train
    panel = build_feature_panel()
    results = walk_forward_train_predict(panel)

    # Load predictions and run extensions
    preds, market, regimes = _load_data()

    # Baseline
    baseline_df = _run_ls(preds)
    baseline_m = _metrics(baseline_df)

    # Anti-Momentum
    am_df, _ = ext1_anti_momentum_features(preds.copy(), market.copy(), regimes.copy())
    am_m = _metrics(am_df)

    # Restore
    mc.FEATURE_COLS = list(FEATURE_COLS)

    return {
        "label": label,
        "ic": results["predicted_return"].corr(results["actual_return"]) if not results.empty else 0,
        "mean_ic": results.groupby(results["date"].dt.to_period("M")).apply(
            lambda g: g["predicted_return"].corr(g["actual_return"])
        ).mean() if not results.empty else 0,
        "baseline": baseline_m,
        "anti_momentum": am_m,
    }


def main():
    r_without = run_variant("WITHOUT Fundamentals (10 features)", use_fundamentals=False)
    r_with = run_variant("WITH Fundamentals (16 features)", use_fundamentals=True)

    print("\n" + "=" * 80)
    print("  A/B COMPARISON: FUNDAMENTAL FEATURES")
    print("=" * 80)

    # ML Combiner signal quality
    print(f"\n  ML Combiner Signal Quality:")
    print(f"  {'Metric':<20s} {'Without':>12s} {'With':>12s} {'Delta':>12s}")
    print(f"  {'-'*56}")
    for key, fmt in [("ic", "{:+.4f}"), ("mean_ic", "{:+.4f}")]:
        v1, v2 = r_without[key], r_with[key]
        d = v2 - v1
        print(f"  {key:<20s} {fmt.format(v1):>12s} {fmt.format(v2):>12s} {fmt.format(d):>12s}")

    # Strategy comparison
    for strat_key, strat_name in [("baseline", "Baseline L/S"), ("anti_momentum", "Anti-Momentum")]:
        print(f"\n  {strat_name}:")
        print(f"  {'Metric':<20s} {'Without':>12s} {'With':>12s} {'Delta':>12s}")
        print(f"  {'-'*56}")
        m1 = r_without[strat_key]
        m2 = r_with[strat_key]
        for key, label, fmt in [
            ("cum", "Cumulative Return", "{:+.1%}"),
            ("ann", "Annualized Return", "{:+.1%}"),
            ("sharpe", "Sharpe Ratio", "{:+.2f}"),
            ("maxdd", "Max Drawdown", "{:.1%}"),
            ("win", "Win Rate", "{:.0%}"),
        ]:
            v1, v2 = m1[key], m2[key]
            d = v2 - v1
            print(f"  {label:<20s} {fmt.format(v1):>12s} {fmt.format(v2):>12s} {fmt.format(d):>12s}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
