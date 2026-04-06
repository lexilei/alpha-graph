"""Transaction cost sensitivity sweep for the ML combiner baseline.

Reruns the long/short backtest at a range of all-in monthly cost assumptions
and reports Sharpe, annualized return, and max drawdown vs cost. The output
identifies the breakeven cost where Sharpe falls below 1.0 and 0.0.

This is the answer to "what if my cost estimate is wrong?" — a question every
quant PM will ask before allocating capital.

Usage:
    python -m alpha_graph.backtest.cost_sensitivity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.backtest.extensions import _load_data, _metrics, _run_ls
from alpha_graph.config import CACHE_DIR, set_global_seeds


# Cost grid in basis points (per month, all-in: spread + commission + borrow)
COST_GRID_BPS = [0, 5, 10, 15, 20, 30, 50, 75, 100]


def run_sweep() -> pd.DataFrame:
    """Run baseline L/S at every cost level on the cached predictions."""
    set_global_seeds()
    preds, _market, _regimes = _load_data()

    rows = []
    for bps in COST_GRID_BPS:
        cost = bps / 10_000.0
        df = _run_ls(preds, cost_per_month=cost)
        m = _metrics(df)
        rows.append({
            "cost_bps_per_month": bps,
            "cost_bps_annualized": bps * 12,
            "cum_return": m["cum"],
            "ann_return": m["ann"],
            "sharpe": m["sharpe"],
            "max_dd": m["maxdd"],
            "win_rate": m["win"],
        })

    return pd.DataFrame(rows)


def find_breakevens(sweep: pd.DataFrame) -> dict:
    """Linearly interpolate the cost where Sharpe crosses 1.0 and 0.0."""
    out = {}
    for threshold in (1.0, 0.5, 0.0):
        # Find the bracket where Sharpe crosses the threshold (going from
        # above to below as cost rises).
        sorted_df = sweep.sort_values("cost_bps_per_month").reset_index(drop=True)
        crossed = None
        for i in range(len(sorted_df) - 1):
            hi = sorted_df.iloc[i]["sharpe"]
            lo = sorted_df.iloc[i + 1]["sharpe"]
            if hi >= threshold and lo < threshold:
                # Linear interp on cost
                c_hi = sorted_df.iloc[i]["cost_bps_per_month"]
                c_lo = sorted_df.iloc[i + 1]["cost_bps_per_month"]
                if hi == lo:
                    crossed = c_lo
                else:
                    crossed = c_hi + (c_lo - c_hi) * (hi - threshold) / (hi - lo)
                break
        out[f"sharpe_{threshold:.1f}"] = crossed

    return out


def main():
    sweep = run_sweep()
    breakevens = find_breakevens(sweep)

    print()
    print("=" * 78)
    print("  COST SENSITIVITY — ML combiner baseline (long/short, top10/bot10)")
    print("=" * 78)
    print(
        f"  {'cost/mo':>10s}  {'cost/yr':>10s}  {'cum':>9s}  {'ann':>9s}  "
        f"{'sharpe':>8s}  {'max DD':>9s}  {'win':>6s}"
    )
    print("-" * 78)
    for _, r in sweep.iterrows():
        print(
            f"  {int(r['cost_bps_per_month']):>5d} bps  "
            f"{int(r['cost_bps_annualized']):>5d} bps  "
            f"{r['cum_return']:>+8.1%}  {r['ann_return']:>+8.1%}  "
            f"{r['sharpe']:>+7.2f}  {r['max_dd']:>+8.1%}  {r['win_rate']:>5.0%}"
        )
    print("-" * 78)
    print("  Breakeven cost (interpolated, bps per month):")
    for k, v in breakevens.items():
        if v is None:
            print(f"    {k}: never crosses (still above threshold at 100 bps)")
        else:
            print(f"    {k}: {v:.1f} bps/mo  ({v * 12:.0f} bps/yr)")
    print("=" * 78)

    out_path = CACHE_DIR / "cost_sensitivity.parquet"
    sweep.to_parquet(out_path, index=False)
    logger.info(f"Saved cost sensitivity sweep to {out_path}")


if __name__ == "__main__":
    main()
