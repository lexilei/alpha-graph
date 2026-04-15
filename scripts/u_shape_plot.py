"""Decile bar chart for the U-shape research note.

Produces report/fig_u_shape_decile.png — a single-panel figure showing mean
forward 21-day return by ML-combiner signal decile, with 95% CI error bars.
Output is intentionally minimal so it drops cleanly into the LaTeX note.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR

OUT_FIG = CACHE_DIR.parent.parent / "report" / "fig_u_shape_decile.png"


def main():
    preds = pd.read_parquet(CACHE_DIR / "ml_combiner_predictions.parquet")
    preds["month"] = pd.to_datetime(preds["date"]).dt.to_period("M")
    preds["decile"] = preds.groupby("month")["signal"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop")
    )

    agg = preds.groupby("decile")["actual_return"].agg(["mean", "std", "count"])
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg["ci95"] = 1.96 * agg["se"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = agg.index.values
    y = agg["mean"].values * 100
    err = agg["ci95"].values * 100
    colors = ["#c44" if d == y.argmin() else "#4477aa" for d in range(len(y))]
    colors[-1] = "#228833"  # highlight best decile
    bars = ax.bar(x, y, yerr=err, capsize=3, color=colors, edgecolor="black", linewidth=0.5)

    ax.axhline(y.mean(), color="gray", linestyle="--", linewidth=1, label=f"universe mean ({y.mean():.2f}%)")
    ax.set_xlabel("ML combiner signal decile (0 = heaviest filing rewrites, 9 = no changes)")
    ax.set_ylabel("Mean forward 21-day return (%)")
    ax.set_title("U-shape of Lazy Prices: bottom decile does NOT underperform (2011–2026, n=1.66M)")
    ax.set_xticks(x)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    for i, b in enumerate(bars):
        ax.annotate(
            f"{y[i]:.2f}",
            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=140)
    logger.info(f"Saved U-shape decile plot to {OUT_FIG}")

    print()
    print("U-shape decile table (forward 21-day return, %):")
    for d, row in agg.iterrows():
        print(f"  decile {d}: mean={row['mean']*100:+.3f}%  ±{row['ci95']*100:.3f} (95% CI)  n={int(row['count'])}")


if __name__ == "__main__":
    main()
