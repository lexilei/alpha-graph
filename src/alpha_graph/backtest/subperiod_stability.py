"""Sub-period stability of Method E across regimes / market eras.

A 14-year aggregate Sharpe can hide a strategy that only worked in one
narrow window. We split the 168-month panel into four roughly-equal
sub-periods and report Method E's Sharpe, annualized return, IC vs forward
return, and IC IR per period.

Sub-periods (chosen for plausible regime breaks, not data-snooped):
  P1: 2012-04 .. 2015-12 — post-GFC recovery, low rates
  P2: 2016-01 .. 2019-12 — Trump tax cut era, low vol
  P3: 2020-01 .. 2022-12 — COVID + inflation regime
  P4: 2023-01 .. 2026-03 — AI boom, rate hikes peak

Output: per-period table + rolling 36-month Sharpe figure
(`report/fig_method_e_rolling_sharpe.png`).

Usage:
    python -m alpha_graph.backtest.subperiod_stability
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from alpha_graph.backtest.attribution import reproduce_method_e
from alpha_graph.backtest.extensions import _load_data
from alpha_graph.backtest.pit_universe import fetch_membership, filter_preds_pit
from alpha_graph.config import CACHE_DIR

SUBPERIODS = [
    ("P1: 2012-04 .. 2015-12 (recovery)",   "2012-04-01", "2016-01-01"),
    ("P2: 2016-01 .. 2019-12 (low-vol)",     "2016-01-01", "2020-01-01"),
    ("P3: 2020-01 .. 2022-12 (COVID/inflation)", "2020-01-01", "2023-01-01"),
    ("P4: 2023-01 .. 2026-03 (AI boom)",     "2023-01-01", "2026-04-01"),
]


def method_e_monthly(preds: pd.DataFrame, cost: float = 0.001) -> pd.DataFrame:
    months = sorted(preds["month"].unique())
    rows = []
    for month in months:
        m = preds[preds["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue
        sorted_m = m.sort_values("signal", ascending=False)
        long_r = float(sorted_m.head(10)["actual_return"].mean())
        rows.append({"month": month.to_timestamp(), "ret": long_r - cost})
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def cross_section_ic(preds: pd.DataFrame) -> pd.DataFrame:
    """Spearman IC of signal vs actual_return per month."""
    out = []
    for month, m in preds.groupby("month"):
        m = m.dropna(subset=["signal", "actual_return"])
        if len(m) < 20:
            continue
        rho, _ = stats.spearmanr(m["signal"], m["actual_return"])
        out.append({"month": month.to_timestamp(), "ic": float(rho), "n": len(m)})
    return pd.DataFrame(out).sort_values("month").reset_index(drop=True)


def summarize(rets: pd.DataFrame, ic_df: pd.DataFrame, start: str, end: str) -> dict:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    block_ret = rets[(rets["month"] >= s) & (rets["month"] < e)]["ret"]
    block_ic = ic_df[(ic_df["month"] >= s) & (ic_df["month"] < e)]["ic"]
    n = len(block_ret)
    if n < 12:
        return {"n_months": n}
    mu = block_ret.mean()
    sd = block_ret.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(12) if sd > 0 else float("nan")
    cum = float((1 + block_ret).prod() - 1)
    ann = (1 + cum) ** (12 / n) - 1
    ic_mean = float(block_ic.mean())
    ic_std = float(block_ic.std(ddof=1))
    icir = (ic_mean / ic_std) * np.sqrt(12) if ic_std > 0 else float("nan")
    # t-stat for Sharpe under iid assumption: t = sharpe * sqrt(n / 12) (per-period)
    sharpe_per = mu / sd if sd > 0 else float("nan")
    t_sharpe = sharpe_per * np.sqrt(n)
    return {
        "n_months": n,
        "ann_return": ann,
        "cum_return": cum,
        "sharpe": sharpe,
        "sharpe_t_stat": t_sharpe,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir_ann": icir,
    }


def rolling_sharpe(rets: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    r = rets.copy().sort_values("month").reset_index(drop=True)
    r["roll_mu"] = r["ret"].rolling(window, min_periods=window // 2).mean()
    r["roll_sd"] = r["ret"].rolling(window, min_periods=window // 2).std(ddof=1)
    r["rolling_sharpe"] = (r["roll_mu"] / r["roll_sd"]) * np.sqrt(12)
    return r[["month", "rolling_sharpe"]]


def plot_rolling(full_roll: pd.DataFrame, pit_roll: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(full_roll["month"], full_roll["rolling_sharpe"], label="Full universe", color="#cc6677", linewidth=1.6)
    ax.plot(pit_roll["month"], pit_roll["rolling_sharpe"], label="PIT universe", color="#117733", linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Sharpe = 1.0")

    # Sub-period bands
    band_colors = ["#4477aa", "#ee6677", "#228833", "#ccbb44"]
    for (label, s, e), color in zip(SUBPERIODS, band_colors):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color=color)
        ax.text(
            pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2,
            ax.get_ylim()[1] if False else 2.5,
            label.split(":")[0], ha="center", va="top", fontsize=8, color=color,
        )

    ax.set_ylabel("Trailing 36-month Sharpe")
    ax.set_xlabel("Month")
    ax.set_title("Method E rolling 36-month Sharpe by universe (full vs PIT)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    logger.info(f"Saved rolling-Sharpe plot to {out_path}")


def main():
    preds, _market, _regimes = _load_data()
    preds["month"] = pd.to_datetime(preds["date"]).dt.to_period("M")

    # Full universe
    rets_full = method_e_monthly(preds)
    ic_full = cross_section_ic(preds)

    # PIT universe
    membership = fetch_membership()
    pit_preds = filter_preds_pit(preds, membership)
    rets_pit = method_e_monthly(pit_preds)
    ic_pit = cross_section_ic(pit_preds)

    # Per-period summaries
    rows = []
    for label, start, end in SUBPERIODS:
        full = summarize(rets_full, ic_full, start, end)
        pit = summarize(rets_pit, ic_pit, start, end)
        rows.append({"period": label, "universe": "full", **full})
        rows.append({"period": label, "universe": "pit", **pit})
    summary_df = pd.DataFrame(rows)

    # Print
    print()
    print("=" * 95)
    print("  SUB-PERIOD STABILITY OF METHOD E")
    print("=" * 95)
    for label, start, end in SUBPERIODS:
        print(f"\n  {label}")
        for u in ("full", "pit"):
            row = summary_df[(summary_df["period"] == label) & (summary_df["universe"] == u)].iloc[0]
            if pd.isna(row.get("sharpe")):
                print(f"    {u.upper():<5s} | n={int(row['n_months'])} months — too short")
                continue
            print(
                f"    {u.upper():<5s} | n={int(row['n_months'])} | "
                f"Sharpe={row['sharpe']:+.2f} (t={row['sharpe_t_stat']:+.2f}) | "
                f"Ann={row['ann_return']*100:+5.1f}% | "
                f"IC={row['ic_mean']:+.4f} | IC IR={row['ic_ir_ann']:+.2f}"
            )
    print()
    print("=" * 95)
    print("  CONSISTENCY CHECK")
    print("=" * 95)
    for u in ("full", "pit"):
        sharpes = summary_df[summary_df["universe"] == u]["sharpe"].dropna().values
        n_pos = int((sharpes > 0).sum())
        n_above_05 = int((sharpes >= 0.5).sum())
        print(
            f"  {u.upper():<5s}: {len(sharpes)} sub-periods, "
            f"{n_pos} with Sharpe > 0, {n_above_05} with Sharpe >= 0.5, "
            f"min={sharpes.min():+.2f}, max={sharpes.max():+.2f}"
        )
    print("=" * 95)

    # Rolling Sharpe figure
    full_roll = rolling_sharpe(rets_full)
    pit_roll = rolling_sharpe(rets_pit)
    plot_rolling(full_roll, pit_roll, CACHE_DIR.parent.parent / "report" / "fig_method_e_rolling_sharpe.png")

    # Save
    summary_df.to_parquet(CACHE_DIR / "method_e_subperiod_stability.parquet", index=False)
    full_roll.to_parquet(CACHE_DIR / "method_e_rolling_sharpe_full.parquet", index=False)
    pit_roll.to_parquet(CACHE_DIR / "method_e_rolling_sharpe_pit.parquet", index=False)
    logger.info("Saved sub-period summary + rolling Sharpe series")


if __name__ == "__main__":
    main()
