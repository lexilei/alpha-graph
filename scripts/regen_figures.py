"""
Regenerate the key report figures from the full 14-year panel.
Replaces the stale 24-month figures in report/.

Usage:
    python scripts/regen_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path

CACHE = Path("data/cache")
OUT = Path("report")
COST_LO = 0.001 * 0.5   # long-only: 5 bps
COST_LS = 0.002          # long/short: 20 bps


def load_preds():
    p = pd.read_parquet(CACHE / "ml_combiner_predictions.parquet")
    p["month"] = p["date"].dt.to_period("M")
    return p


def method_e_returns(preds):
    """Long-only top10."""
    monthly = []
    for month, grp in preds.groupby("month"):
        g = grp.drop_duplicates("ticker", keep="last")
        top10 = g.nlargest(10, "signal")
        r = top10["actual_return"].mean() - COST_LO
        monthly.append({"month": month, "ret": r})
    return pd.DataFrame(monthly).sort_values("month").reset_index(drop=True)


def baseline_ls_returns(preds):
    """Top10/Bot10 L/S baseline."""
    monthly = []
    for month, grp in preds.groupby("month"):
        g = grp.drop_duplicates("ticker", keep="last")
        top10 = g.nlargest(10, "signal")
        bot10 = g.nsmallest(10, "signal")
        long_r = top10["actual_return"].mean()
        short_r = bot10["actual_return"].mean()
        r = long_r - short_r - COST_LS
        monthly.append({"month": month, "ret": r})
    return pd.DataFrame(monthly).sort_values("month").reset_index(drop=True)


def get_spy(start, end):
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if hasattr(spy.columns, "levels"):
        spy.columns = spy.columns.get_level_values(0)
    sm = spy["Close"].resample("ME").last().pct_change().dropna()
    sm = sm.reset_index()
    sm.columns = ["date", "ret"]
    sm["month"] = sm["date"].dt.to_period("M")
    return sm


def main():
    preds = load_preds()
    me = method_e_returns(preds)
    ls = baseline_ls_returns(preds)

    start = me["month"].min().start_time.strftime("%Y-%m-%d")
    end = me["month"].max().end_time.strftime("%Y-%m-%d")
    spy = get_spy(start, end)

    # Align
    df = me.rename(columns={"ret": "ret_e"})
    df = df.merge(ls.rename(columns={"ret": "ret_ls"})[["month", "ret_ls"]], on="month")
    df = df.merge(spy.rename(columns={"ret": "ret_spy"})[["month", "ret_spy"]], on="month")
    df = df.sort_values("month").reset_index(drop=True)

    cum_e = (1 + df["ret_e"]).cumprod()
    cum_ls = (1 + df["ret_ls"]).cumprod()
    cum_spy = (1 + df["ret_spy"]).cumprod()

    x_labels = df["month"].astype(str)

    # ── Fig 1: Cumulative returns (all three) ──────────────
    print("  fig1 ...")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cum_e.values, color="#2563eb", lw=2, label="Method E (long-only top10)")
    ax.plot(cum_spy.values, color="#94a3b8", lw=1.5, label="SPY")
    ax.plot(cum_ls.values, color="#f59e0b", lw=1.5, ls="--", label="L/S baseline (top10/bot10)")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Cumulative Returns: 14-Year Panel (2012-04 to 2026-03)")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ticks = list(range(0, len(x_labels), 12))
    ax.set_xticks(ticks)
    ax.set_xticklabels([x_labels.iloc[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_cumulative_return.png", dpi=150)
    plt.close()

    # ── Fig 2: Monthly returns (Method E) ──────────────────
    print("  fig2 ...")
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#22c55e" if r >= 0 else "#ef4444" for r in df["ret_e"]]
    ax.bar(range(len(df)), df["ret_e"], color=colors, alpha=0.7, width=1.0)
    ax.axhline(df["ret_e"].mean(), color="#2563eb", ls="--", lw=1,
               label=f'mean = {df["ret_e"].mean():.1%}/mo')
    ax.set_ylabel("Monthly return")
    ax.set_title("Method E: Monthly Returns (168 months)")
    ax.legend(fontsize=9)
    ticks = list(range(0, len(df), 12))
    ax.set_xticks(ticks)
    ax.set_xticklabels([x_labels.iloc[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_monthly_returns.png", dpi=150)
    plt.close()

    # ── Fig 3: Monthly IC ─────────────────────────────────
    print("  fig3 ...")
    ic_monthly = []
    for month, grp in preds.groupby("month"):
        g = grp.drop_duplicates("ticker", keep="last")
        if len(g) < 20:
            continue
        ic = g["signal"].corr(g["actual_return"], method="spearman")
        ic_monthly.append({"month": month, "ic": ic})
    ic_df = pd.DataFrame(ic_monthly).sort_values("month")

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#22c55e" if r >= 0 else "#ef4444" for r in ic_df["ic"]]
    ax.bar(range(len(ic_df)), ic_df["ic"], color=colors, alpha=0.7, width=1.0)
    ax.axhline(ic_df["ic"].mean(), color="#2563eb", ls="--", lw=1,
               label=f'mean IC = {ic_df["ic"].mean():.3f}')
    ax.set_ylabel("Rank IC (Spearman)")
    ax.set_title("Monthly Information Coefficient: Signal vs Forward 21-Day Return")
    ax.legend(fontsize=9)
    ticks = list(range(0, len(ic_df), 12))
    ax.set_xticks(ticks)
    ax.set_xticklabels([ic_df.iloc[i]["month"] for i in ticks],
                       rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_monthly_ic.png", dpi=150)
    plt.close()

    # ── Fig 6: Drawdown (Method E vs SPY) ─────────────────
    print("  fig6 ...")
    def drawdown(cum):
        peak = cum.cummax()
        return (cum - peak) / peak

    dd_e = drawdown(cum_e)
    dd_spy = drawdown(cum_spy)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(dd_e)), dd_e.values, 0, color="#dbeafe", alpha=0.7)
    ax.plot(dd_e.values, color="#2563eb", lw=1.5, label="Method E")
    ax.plot(dd_spy.values, color="#94a3b8", lw=1, label="SPY")
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown: Method E vs SPY")
    ax.legend(fontsize=9)
    ticks = list(range(0, len(x_labels), 12))
    ax.set_xticks(ticks)
    ax.set_xticklabels([x_labels.iloc[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_drawdown.png", dpi=150)
    plt.close()

    # ── Fig 7: All 10 methods comparison ──────────────────
    print("  fig7 ...")
    results = pd.read_parquet(CACHE / "improvements_results.parquet")
    print(f"    Methods: {list(results['name'])}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in results.iterrows():
        marker = "o" if row["sharpe"] > 0 else "x"
        color = "#2563eb" if "Long-only" in str(row["name"]) else "#64748b"
        size = 120 if "Long-only" in str(row["name"]) else 60
        ax.scatter(row["maxdd"] * 100, row["sharpe"], s=size, c=color,
                   marker=marker, zorder=3)
        ax.annotate(row["name"], (row["maxdd"] * 100, row["sharpe"]),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(5, 3), textcoords="offset points")
    ax.axhline(0, color="#ef4444", ls="--", lw=0.8)
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Risk-Return Scatter: All 10 Portfolio Methods (14-Year Panel)")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_strategy_comparison.png", dpi=150)
    plt.close()

    print(f"\n  Done. Key figures regenerated in {OUT}/")


if __name__ == "__main__":
    main()
