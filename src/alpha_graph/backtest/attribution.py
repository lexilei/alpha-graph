"""Beta attribution + holdings concentration for Method E.

Tests whether Method E (long-only top10 Lazy Prices) Sharpe 1.08 is real alpha
or disguised market beta + size/quality factor exposure.

Tasks executed (per project review feedback):
  Task 1: Beta-adjust against SPY. Report alpha (annualized), beta, residual
          Sharpe = alpha / std(residual) * sqrt(12), and proper t-stats
          against H_0: alpha = 0.
  Task 2: Holdings concentration check. Frequency of each ticker in top10
          across 168 months, Mag 7 share, top-5 concentration.

If alpha t-stat < 2 OR Mag 7 picks > 30% OR residual Sharpe < 0.3:
  Method E is NOT a deployable alpha — it's a disguised mega-cap beta basket.
  In that case, the value of the finding is purely diagnostic / attribution-based.

Usage:
    python -m alpha_graph.backtest.attribution
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from scipy import stats

from alpha_graph.backtest.extensions import _load_data
from alpha_graph.config import CACHE_DIR, set_global_seeds


COST_PER_MONTH_LONG = 0.001  # 10 bps for long-only (half of L/S since no short leg)
MAG7 = ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA"]


# ---------------------------------------------------------------------------
# Reproduce Method E and capture holdings
# ---------------------------------------------------------------------------

def reproduce_method_e(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Method E exactly as in improvements.py and return:
       - monthly returns DataFrame (month, ret_gross, ret_net)
       - holdings DataFrame (month, ticker)
    """
    months = sorted(preds["month"].unique())
    rows = []
    holdings = []
    for month in months:
        m = preds[preds["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue
        sorted_m = m.sort_values("signal", ascending=False)
        long_pick = sorted_m.head(10)
        long_r = float(long_pick["actual_return"].mean())
        net = long_r - COST_PER_MONTH_LONG
        rows.append({
            "month": month.to_timestamp(),
            "ret_gross": long_r,
            "ret_net": net,
        })
        for t in long_pick["ticker"].tolist():
            holdings.append({"month": str(month), "ticker": t})
    return pd.DataFrame(rows), pd.DataFrame(holdings)


# ---------------------------------------------------------------------------
# Manual OLS (so we don't depend on statsmodels)
# ---------------------------------------------------------------------------

def ols_with_stats(y: np.ndarray, x: np.ndarray) -> dict:
    """Single-variable OLS: y = alpha + beta * x + epsilon.

    Returns dict with alpha, beta, t-stats, p-values, R², residual std.
    Assumes residuals are i.i.d. normal (standard OLS assumption).
    """
    n = len(y)
    assert len(x) == n
    x_mean = x.mean()
    y_mean = y.mean()

    # Slope and intercept
    cov_xy = ((x - x_mean) * (y - y_mean)).sum()
    var_x = ((x - x_mean) ** 2).sum()
    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean

    # Residuals
    y_hat = alpha + beta * x
    residual = y - y_hat
    sse = (residual ** 2).sum()
    sst = ((y - y_mean) ** 2).sum()
    r_squared = 1 - sse / sst

    # Residual std (with ddof=2 for two estimated params)
    resid_var = sse / (n - 2)
    resid_std = np.sqrt(resid_var)

    # Standard errors
    se_beta = np.sqrt(resid_var / var_x)
    se_alpha = np.sqrt(resid_var * (1.0 / n + x_mean ** 2 / var_x))

    # t-statistics
    t_alpha = alpha / se_alpha
    t_beta = beta / se_beta

    # Two-sided p-values from t-distribution with n-2 df
    p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), df=n - 2))
    p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), df=n - 2))

    return {
        "alpha": alpha,
        "beta": beta,
        "se_alpha": se_alpha,
        "se_beta": se_beta,
        "t_alpha": t_alpha,
        "t_beta": t_beta,
        "p_alpha": p_alpha,
        "p_beta": p_beta,
        "r_squared": r_squared,
        "resid_std": resid_std,
        "n": n,
    }


# ---------------------------------------------------------------------------
# SPY benchmark
# ---------------------------------------------------------------------------

def get_spy_monthly_returns(start: str, end: str) -> pd.Series:
    """Download SPY and return monthly total returns indexed by month-end timestamp."""
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if spy.empty:
        raise RuntimeError("SPY download returned empty DataFrame")
    if hasattr(spy.columns, "levels"):
        spy.columns = spy.columns.get_level_values(0)
    monthly = spy["Close"].resample("ME").last().pct_change().dropna()
    return monthly


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    set_global_seeds()

    # Load data and reproduce Method E
    preds, market, _regimes = _load_data()
    me_df, holdings_df = reproduce_method_e(preds)
    me_df = me_df.sort_values("month").reset_index(drop=True)

    # Sanity: reproduce Sharpe 1.08
    raw_sharpe = me_df["ret_net"].mean() / me_df["ret_net"].std() * np.sqrt(12)
    raw_ann = (1 + me_df["ret_net"]).prod() ** (12 / len(me_df)) - 1
    logger.info(f"Reproduced Method E: Sharpe={raw_sharpe:.3f}, AnnRet={raw_ann*100:.1f}%, n={len(me_df)}")

    # Download SPY for the same period
    start = me_df["month"].min().strftime("%Y-%m-%d")
    end = (me_df["month"].max() + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
    spy_monthly = get_spy_monthly_returns(start, end)

    # Align by month: convert both to month-period for safe matching
    me_df["month_period"] = me_df["month"].dt.to_period("M")
    spy_df = spy_monthly.to_frame(name="spy_ret").reset_index()
    spy_df["month_period"] = pd.to_datetime(spy_df["Date"]).dt.to_period("M")

    aligned = me_df.merge(spy_df[["month_period", "spy_ret"]], on="month_period", how="inner")
    logger.info(f"Aligned months for regression: {len(aligned)}")

    if len(aligned) < 24:
        logger.error(f"Only {len(aligned)} aligned months — abort")
        return

    spy_sharpe = aligned["spy_ret"].mean() / aligned["spy_ret"].std() * np.sqrt(12)
    spy_ann = (1 + aligned["spy_ret"]).prod() ** (12 / len(aligned)) - 1

    # ---- Run beta regression: Method E = alpha + beta * SPY ----
    y = aligned["ret_net"].values
    x = aligned["spy_ret"].values
    fit = ols_with_stats(y, x)

    alpha_monthly = fit["alpha"]
    alpha_annual = alpha_monthly * 12  # arithmetic annualization for alpha
    residual_sharpe = alpha_monthly / fit["resid_std"] * np.sqrt(12)

    # ---- Print attribution table ----
    print()
    print("=" * 80)
    print("  TASK 1 — METHOD E BETA ATTRIBUTION (vs SPY, 168-month panel)")
    print("=" * 80)
    print(f"  Months in regression:        {fit['n']}")
    print(f"  Date range:                  {aligned['month'].min().date()} → {aligned['month'].max().date()}")
    print()
    print("  --- Raw performance ---")
    print(f"  Method E raw Sharpe:         {raw_sharpe:+.3f}")
    print(f"  Method E ann return:         {raw_ann*100:+.2f}%")
    print(f"  SPY raw Sharpe:              {spy_sharpe:+.3f}")
    print(f"  SPY ann return:              {spy_ann*100:+.2f}%")
    print()
    print("  --- Beta regression: r_E = α + β · r_SPY + ε ---")
    print(f"  α (monthly):                 {alpha_monthly*100:+.4f}%")
    print(f"  α (annualized, arithmetic):  {alpha_annual*100:+.2f}%")
    print(f"  β (vs SPY):                  {fit['beta']:+.3f}")
    print(f"  R²:                          {fit['r_squared']:.4f}")
    print()
    print("  --- Statistical significance ---")
    print(f"  α standard error (monthly):  {fit['se_alpha']*100:.4f}%")
    print(f"  α t-statistic:               {fit['t_alpha']:+.2f}")
    print(f"  α p-value (two-sided):       {fit['p_alpha']:.4f}")
    print(f"  β t-statistic:               {fit['t_beta']:+.2f}")
    print(f"  β p-value (two-sided):       {fit['p_beta']:.4f}")
    print()
    print("  --- Residual / true alpha analysis ---")
    print(f"  Residual std (monthly):      {fit['resid_std']*100:.3f}%")
    print(f"  Residual Sharpe (α/σ_ε)·√12: {residual_sharpe:+.3f}")
    print()
    print("  --- Verdict thresholds ---")
    if fit["t_alpha"] < 2.0:
        print("  ⚠️  α t-stat < 2 → alpha NOT statistically distinguishable from market exposure")
    else:
        print(f"  ✓  α t-stat ≥ 2 → alpha statistically significant (t={fit['t_alpha']:.2f})")
    if residual_sharpe < 0.3:
        print(f"  ⚠️  Residual Sharpe {residual_sharpe:.2f} < 0.3 → not deployable as standalone alpha")
    else:
        print(f"  ✓  Residual Sharpe {residual_sharpe:.2f} ≥ 0.3 → defensible weak alpha after beta adjustment")
    print("=" * 80)

    # ---- Task 2: Holdings concentration ----
    print()
    print("=" * 80)
    print("  TASK 2 — METHOD E HOLDINGS CONCENTRATION")
    print("=" * 80)
    counts = holdings_df["ticker"].value_counts()
    total_picks = len(holdings_df)
    n_unique = len(counts)
    months_total = holdings_df["month"].nunique()
    print(f"  Total ticker-month picks:    {total_picks}")
    print(f"  Months covered:              {months_total}")
    print(f"  Unique tickers selected:     {n_unique}")
    print(f"  Avg distinct tickers/month:  {total_picks / months_total:.1f}")
    print()
    print("  Top 15 most-frequently-picked tickers:")
    for i, (ticker, n) in enumerate(counts.head(15).items()):
        pct = n / months_total * 100
        is_mag7 = " [Mag 7]" if ticker in MAG7 else ""
        print(f"    {i+1:2d}. {ticker:6s}  {n:3d} months  ({pct:5.1f}% of all months){is_mag7}")
    print()

    # Mag 7 share of total picks
    mag7_picks = counts[counts.index.isin(MAG7)].sum()
    mag7_share = mag7_picks / total_picks * 100
    print(f"  Mag 7 picks share:           {mag7_picks} / {total_picks} = {mag7_share:.1f}%")

    # Top-5 concentration
    top5_picks = counts.head(5).sum()
    top5_share = top5_picks / total_picks * 100
    print(f"  Top 5 ticker concentration:  {top5_share:.1f}%")
    top10_picks = counts.head(10).sum()
    top10_share = top10_picks / total_picks * 100
    print(f"  Top 10 ticker concentration: {top10_share:.1f}%")
    print()
    print("  --- Verdict thresholds ---")
    if mag7_share > 30:
        print(f"  ⚠️  Mag 7 share {mag7_share:.0f}% > 30% → Method E is a disguised mega-cap quality basket")
    else:
        print(f"  ✓  Mag 7 share {mag7_share:.0f}% ≤ 30% → not over-concentrated in Mag 7")
    if top5_share > 30:
        print(f"  ⚠️  Top-5 concentration {top5_share:.0f}% > 30% → too few names driving the result")
    else:
        print(f"  ✓  Top-5 concentration {top5_share:.0f}% ≤ 30% → reasonably diversified")
    print("=" * 80)

    # ---- Save outputs ----
    out_path = CACHE_DIR / "method_e_attribution.parquet"
    pd.DataFrame({
        "raw_sharpe": [raw_sharpe],
        "raw_ann_return": [raw_ann],
        "spy_sharpe": [spy_sharpe],
        "spy_ann_return": [spy_ann],
        "alpha_monthly": [alpha_monthly],
        "alpha_annual": [alpha_annual],
        "beta": [fit["beta"]],
        "r_squared": [fit["r_squared"]],
        "alpha_t_stat": [fit["t_alpha"]],
        "alpha_p_value": [fit["p_alpha"]],
        "residual_sharpe": [residual_sharpe],
        "n_months": [fit["n"]],
        "mag7_share_pct": [mag7_share],
        "top5_concentration_pct": [top5_share],
    }).to_parquet(out_path, index=False)
    holdings_df.to_parquet(CACHE_DIR / "method_e_holdings.parquet", index=False)
    logger.info(f"Saved attribution to {out_path}")
    logger.info(f"Saved holdings to {CACHE_DIR / 'method_e_holdings.parquet'}")


if __name__ == "__main__":
    main()
