"""
Interview-prep diagnostic plots for Alpha-Graph.
Generates figures into report/diag_*.png.

Usage:
    python -m scripts.diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CACHE = Path("data/cache")
OUT = Path("report")

# ── colours ──────────────────────────────────────────────────
C_MAIN = "#2563eb"
C_SPY = "#94a3b8"
C_ZERO = "#ef4444"
C_FILL = "#dbeafe"
C_BAD = "#fca5a5"


def load():
    preds = pd.read_parquet(CACHE / "ml_combiner_predictions.parquet")
    preds["month"] = preds["date"].dt.to_period("M")
    holdings = pd.read_parquet(CACHE / "method_e_holdings.parquet")
    sector_map = pd.read_parquet(CACHE / "sector_map.parquet")
    market = pd.read_parquet(CACHE / "market_data.parquet")
    lp = pd.read_parquet(CACHE / "lazy_prices_signal_10K.parquet")
    return preds, holdings, sector_map, market, lp


# ═══════════════════════════════════════════════════════════════
# 1. Rolling 12-month Sharpe for Method E
# ═══════════════════════════════════════════════════════════════
def plot_rolling_sharpe(preds):
    print("  [1/6] Rolling 12-month Sharpe ...")
    COST = 0.001 * 0.5  # 10 bps, halved for long-only

    monthly = []
    for month, grp in preds.groupby("month"):
        g = grp.drop_duplicates("ticker", keep="last")
        top10 = g.nlargest(10, "signal")
        r = top10["actual_return"].mean() - COST
        monthly.append({"month": month, "ret": r})

    df = pd.DataFrame(monthly).sort_values("month")
    df["roll_mean"] = df["ret"].rolling(12).mean()
    df["roll_std"] = df["ret"].rolling(12).std()
    df["roll_sharpe"] = (df["roll_mean"] / df["roll_std"]) * np.sqrt(12)
    df = df.dropna()

    x = df["month"].astype(str)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(x)), df["roll_sharpe"], 0,
                     where=df["roll_sharpe"] >= 0, color=C_FILL, alpha=0.7)
    ax.fill_between(range(len(x)), df["roll_sharpe"], 0,
                     where=df["roll_sharpe"] < 0, color=C_BAD, alpha=0.5)
    ax.plot(range(len(x)), df["roll_sharpe"], color=C_MAIN, lw=1.5)
    ax.axhline(0, color=C_ZERO, lw=0.8, ls="--")
    ax.axhline(df["roll_sharpe"].mean(), color="gray", lw=0.8, ls=":",
               label=f'mean = {df["roll_sharpe"].mean():.2f}')
    ticks = list(range(0, len(x), 12))
    ax.set_xticks(ticks)
    ax.set_xticklabels([x.iloc[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rolling 12-mo Sharpe")
    ax.set_title("Method E: Rolling 12-Month Sharpe Ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "diag_rolling_sharpe.png", dpi=150)
    plt.close()
    return df


# ═══════════════════════════════════════════════════════════════
# 2. Decile forward-return heatmap over time
# ═══════════════════════════════════════════════════════════════
def plot_decile_heatmap(preds):
    print("  [2/6] Decile heatmap over time ...")
    preds = preds.copy()
    preds["year"] = preds["date"].dt.year

    rows = []
    for (year, month), grp in preds.groupby(["year", "month"]):
        g = grp.drop_duplicates("ticker", keep="last")
        g["decile"] = pd.qcut(g["signal"], 10, labels=False, duplicates="drop")
        for d, dg in g.groupby("decile"):
            rows.append({"year": year, "decile": d,
                         "fwd_ret": dg["actual_return"].mean()})

    df = pd.DataFrame(rows)
    pivot = df.groupby(["year", "decile"])["fwd_ret"].mean().unstack("decile")

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values.T, aspect="auto", cmap="RdYlGn",
                   vmin=-0.01, vmax=0.04)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"D{i}" for i in range(10)])
    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot.index.astype(int), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Signal Decile (0=lowest, 9=highest)")
    ax.set_xlabel("Year")
    ax.set_title("Avg Forward 21-Day Return by Signal Decile and Year")
    fig.colorbar(im, ax=ax, label="Avg monthly return", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "diag_decile_heatmap.png", dpi=150)
    plt.close()
    return pivot


# ═══════════════════════════════════════════════════════════════
# 3. Survivorship bias check
# ═══════════════════════════════════════════════════════════════
def plot_survivorship(preds, holdings):
    print("  [3/6] Survivorship bias check ...")
    COST = 0.001 * 0.5

    # Heuristic: tickers with first filing after 2015 are likely post-2015 joiners
    first_date = preds.groupby("ticker")["date"].min().reset_index()
    first_date.columns = ["ticker", "first_date"]
    post2015 = set(first_date.loc[first_date["first_date"].dt.year > 2015, "ticker"])
    pre2015 = set(first_date["ticker"]) - post2015

    print(f"    Pre-2015 tickers: {len(pre2015)}, Post-2015 joiners dropped: {len(post2015)}")

    preds_filtered = preds[preds["ticker"].isin(pre2015)].copy()

    results = {}
    for label, p in [("Full universe", preds), ("Pre-2015 only", preds_filtered)]:
        monthly = []
        for month, grp in p.groupby("month"):
            g = grp.drop_duplicates("ticker", keep="last")
            top10 = g.nlargest(10, "signal")
            r = top10["actual_return"].mean() - COST
            monthly.append(r)
        rets = pd.Series(monthly)
        cum = (1 + rets).cumprod()
        sharpe = rets.mean() / rets.std() * np.sqrt(12)
        results[label] = {"cum": cum, "sharpe": sharpe, "ann": (cum.iloc[-1]) ** (12 / len(cum)) - 1}

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, d in results.items():
        style = {"Full universe": (C_MAIN, "-"), "Pre-2015 only": ("#f59e0b", "--")}[label]
        ax.plot(d["cum"].values, color=style[0], ls=style[1], lw=1.5,
                label=f'{label}: Sharpe {d["sharpe"]:.2f}, Ann {d["ann"]:.1%}')
    ax.set_ylabel("Cumulative return (growth of $1)")
    ax.set_title("Method E: Survivorship Bias Check (drop post-2015 S&P 500 joiners)")
    ax.legend(fontsize=8)
    ax.set_xlabel("Month index")
    fig.tight_layout()
    fig.savefig(OUT / "diag_survivorship.png", dpi=150)
    plt.close()
    return results


# ═══════════════════════════════════════════════════════════════
# 4. Sector composition over time
# ═══════════════════════════════════════════════════════════════
def plot_sector_composition(holdings, sector_map):
    print("  [4/6] Sector composition over time ...")
    df = holdings.merge(sector_map, on="ticker", how="left")
    ct = df.groupby(["month", "sector"]).size().unstack(fill_value=0)
    ct = ct.div(ct.sum(axis=1), axis=0)  # normalize to fractions

    fig, ax = plt.subplots(figsize=(12, 5))
    ct.plot.area(ax=ax, stacked=True, alpha=0.8, lw=0)
    ax.set_ylabel("Share of top-10 picks")
    ax.set_title("Method E: Sector Composition Over Time")
    ax.legend(fontsize=6, loc="upper left", ncol=3, framealpha=0.8)
    ticks = list(range(0, len(ct), 12))
    ax.set_xticks(ticks)
    ax.set_xticklabels([ct.index[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "diag_sector_composition.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 5. Ticker turnover
# ═══════════════════════════════════════════════════════════════
def plot_turnover(holdings):
    print("  [5/6] Ticker turnover ...")
    months = sorted(holdings["month"].unique())
    turnover = []
    for i in range(1, len(months)):
        prev = set(holdings.loc[holdings["month"] == months[i - 1], "ticker"])
        curr = set(holdings.loc[holdings["month"] == months[i], "ticker"])
        if len(prev) == 0:
            continue
        changed = len(curr - prev) / max(len(curr), 1)
        turnover.append({"month": months[i], "turnover": changed})

    df = pd.DataFrame(turnover)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(df)), df["turnover"], color=C_MAIN, alpha=0.6, width=1.0)
    ax.axhline(df["turnover"].mean(), color=C_ZERO, ls="--", lw=1,
               label=f'mean = {df["turnover"].mean():.1%}')
    ax.set_ylabel("Fraction of top-10 replaced")
    ax.set_title("Method E: Monthly Ticker Turnover")
    ticks = list(range(0, len(df), 12))
    ax.set_xticks(ticks)
    ax.set_xticklabels([df.iloc[i]["month"] for i in ticks],
                       rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "diag_turnover.png", dpi=150)
    plt.close()
    return df


# ═══════════════════════════════════════════════════════════════
# 6. Rolling alpha vs SPY
# ═══════════════════════════════════════════════════════════════
def plot_rolling_alpha(preds, market):
    print("  [6/6] Rolling 24-month alpha vs SPY ...")
    COST = 0.001 * 0.5

    # Method E monthly returns
    monthly_e = []
    for month, grp in preds.groupby("month"):
        g = grp.drop_duplicates("ticker", keep="last")
        top10 = g.nlargest(10, "signal")
        r = top10["actual_return"].mean() - COST
        monthly_e.append({"month": month, "ret_e": r})

    df_e = pd.DataFrame(monthly_e).sort_values("month")

    # SPY monthly returns via yfinance
    import yfinance as yf
    start = df_e["month"].min().start_time.strftime("%Y-%m-%d")
    end = df_e["month"].max().end_time.strftime("%Y-%m-%d")
    spy_raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if hasattr(spy_raw.columns, "levels"):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_m = spy_raw["Close"].resample("ME").last().pct_change().dropna()
    spy_m = spy_m.reset_index()
    spy_m.columns = ["date", "ret_spy"]
    spy_m["month"] = spy_m["date"].dt.to_period("M")

    df = df_e.merge(spy_m[["month", "ret_spy"]], on="month").sort_values("month").reset_index(drop=True)

    # Rolling 24-month OLS: ret_e = alpha + beta * ret_spy
    window = 24
    alphas, betas, t_stats = [], [], []
    for i in range(window, len(df)):
        y = df["ret_e"].iloc[i - window:i].values
        x = df["ret_spy"].iloc[i - window:i].values
        X = np.column_stack([np.ones(window), x])
        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ coef
            se = np.sqrt(np.sum(resid**2) / (window - 2) / np.sum((x - x.mean())**2))
            se_alpha = np.sqrt(np.sum(resid**2) / (window - 2) *
                               (1/window + x.mean()**2 / np.sum((x - x.mean())**2)))
            alphas.append(coef[0] * 12)  # annualize
            betas.append(coef[1])
            t_stats.append(coef[0] / se_alpha if se_alpha > 0 else 0)
        except Exception:
            alphas.append(np.nan)
            betas.append(np.nan)
            t_stats.append(np.nan)

    x_axis = df["month"].iloc[window:].astype(str)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Alpha
    ax = axes[0]
    ax.fill_between(range(len(alphas)), alphas, 0,
                     where=[a >= 0 for a in alphas], color=C_FILL, alpha=0.7)
    ax.fill_between(range(len(alphas)), alphas, 0,
                     where=[a < 0 for a in alphas], color=C_BAD, alpha=0.5)
    ax.plot(range(len(alphas)), alphas, color=C_MAIN, lw=1.5)
    ax.axhline(0, color=C_ZERO, lw=0.8, ls="--")
    ax.set_ylabel("Rolling 24-mo alpha\n(annualized)")
    ax.set_title("Method E: Rolling Alpha & t-stat vs SPY")

    # t-stat
    ax = axes[1]
    ax.bar(range(len(t_stats)), t_stats, color=C_MAIN, alpha=0.6, width=1.0)
    ax.axhline(1.96, color="#22c55e", ls="--", lw=0.8, label="t = 1.96")
    ax.axhline(-1.96, color=C_ZERO, ls="--", lw=0.8)
    ax.set_ylabel("Alpha t-stat")
    ticks = list(range(0, len(x_axis), 12))
    ax.set_xticks(ticks)
    ax.set_xticklabels([x_axis.iloc[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "diag_rolling_alpha.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
def main():
    print("Loading data ...")
    preds, holdings, sector_map, market, lp = load()

    print("Generating diagnostic plots ...")
    plot_rolling_sharpe(preds)
    plot_decile_heatmap(preds)
    plot_survivorship(preds, holdings)
    plot_sector_composition(holdings, sector_map)
    plot_turnover(holdings)
    plot_rolling_alpha(preds, market)

    print(f"\nDone. Figures saved to {OUT}/diag_*.png")


if __name__ == "__main__":
    main()
