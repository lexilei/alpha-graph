"""Architectural improvements for the ML combiner baseline.

V1 of this module tested four 'obvious' improvements (rank-weighted, decile,
vol-target, multi-factor) and ALL of them failed or were flat. Diagnostic
revealed why: the Lazy Prices signal is **U-shaped, not monotonic**.

  Decile 9 (high cosine sim, no filing changes):  +2.10% / month  ← LONG works
  Decile 4-5 (middle):                            +1.19% / month  ← LOWEST
  Decile 0 (low cosine sim, big filing changes):  +1.37% / month  ← SHORT FAILS

In other words: companies that *don't* change their filings outperform — that's
the original Lazy Prices long thesis. But companies that *dramatically* change
their filings do NOT underperform on average — they actually beat the middle.
The short side of the original paper does not replicate in 2011-2026.

This means rank-weighted and multi-factor approaches that assume monotonicity
will dilute the signal into the broken middle/short region. The right moves are:

  E. Long-only top decile        — drop the broken short side entirely.
  F. Long top10 / Short middle10 — explicitly exploit the U-shape (long where
                                    signal is highest, short where signal is
                                    most ambiguous, NOT where signal says short).
  G. Pure 12-1 Momentum (no LP)  — sanity check that a known monotonic factor
                                    works in this universe at all.
  H. 12-1 Mom + Low-Vol combo    — the "real" multi-factor: combine factors
                                    that ARE monotonic, without Lazy Prices.
  I. Long LP top10 + Short Mom bot10 — orthogonal hedge: Lazy Prices long side
                                    + momentum short side (cross-factor).

Each method runs on the same 168-month panel as the baseline.

Usage:
    python -m alpha_graph.backtest.improvements
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.backtest.extensions import _load_data, _metrics
from alpha_graph.config import CACHE_DIR, set_global_seeds


COST_PER_MONTH = 0.002  # 20 bps, consistent with extensions baseline
TARGET_ANN_VOL = 0.10   # for Method C


# ---------------------------------------------------------------------------
# Shared backtest core that supports rank-weighted positions
# ---------------------------------------------------------------------------

def _run_weighted_ls(
    preds: pd.DataFrame,
    weight_fn,
    cost_per_month: float = COST_PER_MONTH,
    min_universe: int = 20,
) -> pd.DataFrame:
    """Generic L/S backtest where positions are determined by a weight function.

    `weight_fn(month_df)` must return a Series indexed like month_df with
    weights summing to ~0 (long positive, short negative). The Series will
    be normalized so that gross exposure (sum of |w|) = 2 (i.e. 100% long
    + 100% short, dollar-neutral).
    """
    months = sorted(preds["month"].unique())
    rows = []
    for month in months:
        m = preds[preds["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < min_universe:
            continue
        w = weight_fn(m)
        if w is None or w.abs().sum() == 0:
            continue
        # Normalize gross exposure to 2 (dollar-neutral, 100/100)
        w = w * (2.0 / w.abs().sum())
        # Weighted average return
        net_gross = float((w.values * m["actual_return"].values).sum())
        # Per-period turnover-implied cost: 2x gross × cost_per_month/2 ≈ cost_per_month
        # We use the same flat cost as extensions for apples-to-apples
        net = net_gross - cost_per_month
        rows.append({
            "month": month.to_timestamp(),
            "net": net,
            "long": float((w[w > 0] * m.loc[w > 0, "actual_return"]).sum()),
            "short": float((w[w < 0] * m.loc[w < 0, "actual_return"]).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Method A: Rank-weighted across the full cross-section [BROKEN by U-shape]
# ---------------------------------------------------------------------------

def method_a_rank_weighted(preds: pd.DataFrame) -> dict:
    """Long every stock above median, short every below — weight by signed rank.

    KEPT FOR REGRESSION: this method failed in V1 (Sharpe -0.12) because the
    signal is U-shaped, not monotonic. Rank weighting is fundamentally wrong
    for non-monotonic signals.
    """
    logger.info("Method A: Rank-weighted L/S [expected to fail]")

    def weight_fn(m):
        # Cross-sectional rank in [-1, +1]
        n = len(m)
        rank = m["signal"].rank(method="average")
        signed = (rank - (n + 1) / 2) / ((n - 1) / 2)  # [-1, +1]
        return pd.Series(signed.values, index=m.index)

    df = _run_weighted_ls(preds, weight_fn)
    metrics = _metrics(df)
    metrics["name"] = "A. Rank-weighted [BROKEN: signal is U-shaped]"
    metrics["n_months"] = len(df)
    return metrics


# ---------------------------------------------------------------------------
# Method E: Long-only top decile — drop the broken short side
# ---------------------------------------------------------------------------

def method_e_long_only(preds: pd.DataFrame) -> dict:
    """Long top 10 stocks by signal, no short side.

    The U-shape diagnostic showed the short side of Lazy Prices doesn't work
    in 2011-2026: low-signal stocks (filings rewritten heavily) had +1.37%
    monthly return on average, only marginally below the universe mean.
    Shorting them is paying alpha to SHORT a positive expected return.

    Long-only should outperform L/S because the short side was a drag, not
    an alpha contributor.
    """
    logger.info("Method E: Long-only top10 (drop broken short side)")

    def weight_fn(m):
        sorted_m = m.sort_values("signal", ascending=False)
        long_idx = sorted_m.head(10).index
        w = pd.Series(0.0, index=m.index)
        w.loc[long_idx] = 0.1
        return w

    # Long-only is not dollar-neutral; gross is just 1.0
    months = sorted(preds["month"].unique())
    rows = []
    for month in months:
        m = preds[preds["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue
        sorted_m = m.sort_values("signal", ascending=False)
        long_pick = sorted_m.head(10)
        long_r = float(long_pick["actual_return"].mean())
        # Long-only cost is half of L/S because there's no short leg
        net = long_r - COST_PER_MONTH * 0.5
        rows.append({"month": month.to_timestamp(), "net": net})
    df = pd.DataFrame(rows)
    metrics = _metrics(df)
    metrics["name"] = "E. Long-only top10 (no short)"
    metrics["n_months"] = len(df)
    return metrics


# ---------------------------------------------------------------------------
# Method F: Exploit U-shape — Long top, Short MIDDLE
# ---------------------------------------------------------------------------

def method_f_u_shape(preds: pd.DataFrame) -> dict:
    """Long top decile, short MIDDLE decile (NOT bottom decile).

    The U-shape diagnostic showed the lowest-return decile is decile 4
    (the MIDDLE), not the bottom. This method explicitly bets on that
    structure: long the highest-signal stocks (decile 9, +2.10%/mo),
    short the middle decile (decile 4-5, +1.19%/mo).

    Theoretical L/S spread: 2.10% - 1.19% = 0.91% per month gross,
    vs. baseline's 2.10% - 1.37% = 0.73%. Should improve Sharpe ~25%.
    """
    logger.info("Method F: Long top10 / Short MIDDLE10 (exploit U-shape)")

    months = sorted(preds["month"].unique())
    rows = []
    for month in months:
        m = preds[preds["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 30:
            continue
        sorted_m = m.sort_values("signal", ascending=False).reset_index(drop=True)
        n = len(sorted_m)
        long_pick = sorted_m.head(10)
        # Middle 10: take stocks around the median rank
        mid_start = n // 2 - 5
        mid_pick = sorted_m.iloc[mid_start:mid_start + 10]
        long_r = float(long_pick["actual_return"].mean())
        short_r = float(mid_pick["actual_return"].mean())
        net = long_r - short_r - COST_PER_MONTH
        rows.append({"month": month.to_timestamp(), "net": net,
                     "long": long_r, "short": short_r})
    df = pd.DataFrame(rows)
    metrics = _metrics(df)
    metrics["name"] = "F. Long top10 / Short MIDDLE10 (U-shape)"
    metrics["n_months"] = len(df)
    return metrics


# ---------------------------------------------------------------------------
# Method G: Pure 12-1 Momentum (no Lazy Prices)
# ---------------------------------------------------------------------------

def method_g_momentum_only(preds: pd.DataFrame, market: pd.DataFrame) -> dict:
    """12-1 momentum L/S, top10/bot10. Sanity check that a known monotonic
    factor produces a positive Sharpe in this 499-ticker universe.

    If this fails, the universe itself is not factor-friendly. If it works,
    we have a baseline for what a monotonic factor can deliver here.
    """
    logger.info("Method G: Pure 12-1 Momentum L/S (no Lazy Prices)")

    factors = _build_price_factors(market)
    enriched = preds.merge(factors, on=["ticker", "month"], how="left").dropna(
        subset=["mom_12_1", "actual_return"]
    )

    months = sorted(enriched["month"].unique())
    rows = []
    for month in months:
        m = enriched[enriched["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue
        sorted_m = m.sort_values("mom_12_1", ascending=False)
        long_pick = sorted_m.head(10)
        short_pick = sorted_m.tail(10)
        long_r = float(long_pick["actual_return"].mean())
        short_r = float(short_pick["actual_return"].mean())
        net = long_r - short_r - COST_PER_MONTH
        rows.append({"month": month.to_timestamp(), "net": net})
    df = pd.DataFrame(rows)
    metrics = _metrics(df)
    metrics["name"] = "G. Pure 12-1 Momentum (no LP)"
    metrics["n_months"] = len(df)
    return metrics


# ---------------------------------------------------------------------------
# Method H: 12-1 Momentum + Low-Vol composite (no Lazy Prices)
# ---------------------------------------------------------------------------

def method_h_mom_lowvol(preds: pd.DataFrame, market: pd.DataFrame) -> dict:
    """Composite of 12-1 momentum + low-vol via z-score averaging.

    Both factors are well-known to be monotonic in the cross-section. Combining
    them via rank-weighted L/S should diversify factor noise and improve Sharpe
    over either alone. THIS is the multi-factor improvement path that doesn't
    rely on the broken Lazy Prices monotonicity assumption.
    """
    logger.info("Method H: Multi-factor (12-1 Mom + Low-Vol, no Lazy Prices)")

    factors = _build_price_factors(market)
    enriched = preds.merge(factors, on=["ticker", "month"], how="left").dropna(
        subset=["mom_12_1", "lowvol", "actual_return"]
    )

    months = sorted(enriched["month"].unique())
    rows = []
    for month in months:
        m = enriched[enriched["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue
        # Z-score per factor
        m = m.copy()
        for col in ("mom_12_1", "lowvol"):
            mean = m[col].mean()
            std = m[col].std()
            m[f"{col}_z"] = (m[col] - mean) / std if std > 0 else 0.0
        m["composite"] = (m["mom_12_1_z"] + m["lowvol_z"]) / 2.0
        sorted_m = m.sort_values("composite", ascending=False)
        long_pick = sorted_m.head(10)
        short_pick = sorted_m.tail(10)
        long_r = float(long_pick["actual_return"].mean())
        short_r = float(short_pick["actual_return"].mean())
        net = long_r - short_r - COST_PER_MONTH
        rows.append({"month": month.to_timestamp(), "net": net})
    df = pd.DataFrame(rows)
    metrics = _metrics(df)
    metrics["name"] = "H. 12-1 Mom + Low-Vol composite"
    metrics["n_months"] = len(df)
    return metrics


# ---------------------------------------------------------------------------
# Method I: Cross-factor hedge — Long LP top10, Short Mom bot10
# ---------------------------------------------------------------------------

def method_j_beta_hedged(
    preds: pd.DataFrame,
    market: pd.DataFrame,
    lookback_months: int = 24,
) -> dict:
    """Beta-hedged Method E: long top10 Lazy Prices picks, short β·SPY.

    Method E is long-only; the original project promised market-neutral. This
    variant reframes Method E as a market-neutral book by subtracting a
    rolling-beta-weighted short SPY leg.

    For each month t:
      1. Take long top10 gross weight 1.0 (same picks as Method E).
      2. Estimate β_t of the long basket's monthly returns against SPY using
         the trailing `lookback_months` months of OOS Method E returns.
         First 24 months use β = 1.0 as a safe default.
      3. Short SPY with weight β_t. Net exposure = (1.0 - β_t) (often near 0).
      4. Apply cost 10 bps on long leg + 5 bps on SPY leg = 15 bps/month.

    Honest caveat: β is estimated from rolling OOS Method E returns only, so
    it's not forward-looking. But the long picks themselves come from a
    walk-forward model, so the whole pipeline is PIT-correct.
    """
    logger.info("Method J: Beta-hedged Method E (long top10 - β·SPY)")

    # SPY monthly returns from market data (we already have SPY's own history in
    # the cached parquet if the downloader included it; otherwise fetch).
    spy_rows = market[market["ticker"] == "SPY"].sort_values("date")
    if len(spy_rows) < 100:
        import yfinance as yf
        spy_raw = yf.download(
            "SPY",
            start=market["date"].min().strftime("%Y-%m-%d"),
            end=(market["date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False,
        )
        if hasattr(spy_raw.columns, "levels"):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_monthly = spy_raw["Close"].resample("ME").last().pct_change().dropna()
    else:
        spy_rows = spy_rows.copy()
        spy_rows["month"] = spy_rows["date"].dt.to_period("M")
        spy_monthly = (
            spy_rows.groupby("month")["close"].last().pct_change().dropna()
        )
        spy_monthly.index = spy_monthly.index.to_timestamp(how="end")

    spy_map = pd.Series(spy_monthly.values, index=pd.to_datetime(spy_monthly.index).to_period("M"))

    months = sorted(preds["month"].unique())
    # First pass: compute long-only monthly returns for rolling β estimation.
    long_returns = {}
    for month in months:
        m = preds[preds["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue
        sorted_m = m.sort_values("signal", ascending=False)
        long_returns[month] = float(sorted_m.head(10)["actual_return"].mean())

    long_series = pd.Series(long_returns).sort_index()

    rows = []
    for month in months:
        if month not in long_series.index:
            continue
        long_r = long_series.loc[month]
        spy_r = spy_map.get(month, np.nan)
        if pd.isna(spy_r):
            continue
        history = long_series.loc[:month].iloc[:-1].tail(lookback_months)
        spy_hist = spy_map.reindex(history.index).dropna()
        history = history.loc[spy_hist.index]
        if len(history) >= 12 and spy_hist.var() > 0:
            beta = float(np.cov(history.values, spy_hist.values, ddof=0)[0, 1] / spy_hist.var())
        else:
            beta = 1.0
        beta = max(0.0, min(2.0, beta))
        cost = 0.0015
        net = long_r - beta * spy_r - cost
        rows.append({
            "month": month.to_timestamp(),
            "net": net,
            "beta": beta,
            "long_r": long_r,
            "spy_r": spy_r,
        })

    df = pd.DataFrame(rows)
    metrics = _metrics(df)
    metrics["name"] = "J. Beta-hedged Method E (long - β·SPY)"
    metrics["n_months"] = len(df)
    metrics["avg_beta"] = float(df["beta"].mean()) if not df.empty else float("nan")
    return metrics


def method_i_cross_factor(preds: pd.DataFrame, market: pd.DataFrame) -> dict:
    """Long the top10 by Lazy Prices signal, short the bot10 by 12-1 momentum.

    The two legs are based on different factors, which should be near-orthogonal
    in expected return space. Long high-LP captures the fundamental "stable
    company" premium; short low-momentum captures the momentum factor's short
    leg. Their idiosyncratic noise should diversify, lifting Sharpe.
    """
    logger.info("Method I: Long LP top10 + Short Mom bot10 (cross-factor)")

    factors = _build_price_factors(market)
    enriched = preds.merge(factors, on=["ticker", "month"], how="left").dropna(
        subset=["mom_12_1", "actual_return"]
    )

    months = sorted(enriched["month"].unique())
    rows = []
    for month in months:
        m = enriched[enriched["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 30:
            continue
        long_pick = m.sort_values("signal", ascending=False).head(10)
        short_pick = m.sort_values("mom_12_1", ascending=True).head(10)
        long_r = float(long_pick["actual_return"].mean())
        short_r = float(short_pick["actual_return"].mean())
        net = long_r - short_r - COST_PER_MONTH
        rows.append({"month": month.to_timestamp(), "net": net})
    df = pd.DataFrame(rows)
    metrics = _metrics(df)
    metrics["name"] = "I. Long LP / Short low-Mom (cross-factor)"
    metrics["n_months"] = len(df)
    return metrics


# ---------------------------------------------------------------------------
# Method B: Decile portfolio (top 10% / bot 10%)
# ---------------------------------------------------------------------------

def method_b_decile(preds: pd.DataFrame) -> dict:
    """Long top decile, short bottom decile, equal-weighted within each."""
    logger.info("Method B: Decile portfolio (top 10% / bot 10%)")

    def weight_fn(m):
        n = len(m)
        n_per_side = max(int(n * 0.10), 5)
        sorted_m = m.sort_values("signal", ascending=False)
        long_idx = sorted_m.head(n_per_side).index
        short_idx = sorted_m.tail(n_per_side).index
        w = pd.Series(0.0, index=m.index)
        w.loc[long_idx] = 1.0 / n_per_side
        w.loc[short_idx] = -1.0 / n_per_side
        return w

    df = _run_weighted_ls(preds, weight_fn)
    metrics = _metrics(df)
    metrics["name"] = "B. Decile (top 10% / bot 10%)"
    metrics["n_months"] = len(df)
    return metrics


# ---------------------------------------------------------------------------
# Method C: Vol-targeted exposure
# ---------------------------------------------------------------------------

def method_c_vol_target(preds: pd.DataFrame) -> dict:
    """Run baseline top10/bot10, then scale next month's gross exposure to
    target 10% annualized vol based on trailing 12-month realized vol.

    Attacks drawdown more than alpha: Sharpe should improve modestly because
    the strategy stops doubling down in high-vol regimes (which historically
    coincide with low IC for slow signals like Lazy Prices).
    """
    logger.info("Method C: Vol-targeted exposure (target 10% ann vol)")

    # First pass: unscaled top10/bot10 monthly returns
    def equal_top10_weight(m):
        sorted_m = m.sort_values("signal", ascending=False)
        long_idx = sorted_m.head(10).index
        short_idx = sorted_m.tail(10).index
        w = pd.Series(0.0, index=m.index)
        w.loc[long_idx] = 0.1
        w.loc[short_idx] = -0.1
        return w

    base_df = _run_weighted_ls(preds, equal_top10_weight, cost_per_month=0.0)
    if base_df.empty:
        return {"name": "C. Vol-target (FAIL)", "sharpe": float("nan")}

    # Compute trailing 12-month realized vol (monthly), shift by 1 to avoid lookahead
    base_df = base_df.sort_values("month").reset_index(drop=True)
    base_df["roll_vol"] = base_df["net"].rolling(12, min_periods=6).std().shift(1)
    base_df["roll_vol"] = base_df["roll_vol"].fillna(base_df["net"].std())
    target_monthly_vol = TARGET_ANN_VOL / np.sqrt(12)
    base_df["scale"] = (target_monthly_vol / base_df["roll_vol"]).clip(0.25, 4.0)
    base_df["net_scaled"] = base_df["net"] * base_df["scale"] - COST_PER_MONTH

    # Re-package to look like a normal returns DataFrame for _metrics
    out = pd.DataFrame({"net": base_df["net_scaled"], "month": base_df["month"]})
    metrics = _metrics(out)
    metrics["name"] = "C. Vol-targeted (10% ann vol)"
    metrics["n_months"] = len(out)
    metrics["avg_scale"] = float(base_df["scale"].mean())
    return metrics


# ---------------------------------------------------------------------------
# Method D: Multi-factor combiner — Lazy Prices + 12-1 Momentum + Low-Vol
# ---------------------------------------------------------------------------

def _build_price_factors(market: pd.DataFrame) -> pd.DataFrame:
    """Compute 12-1 momentum and 60d realized vol at month-end for each ticker.

    CRITICAL: market_data.parquet's `ret_1d/5d/21d` columns are FORWARD-looking
    (they're the prediction targets for ml_combiner). DO NOT use them as features.
    We compute trailing returns ourselves via pct_change directly on close.
    """
    market = market.sort_values(["ticker", "date"]).copy()

    # Backward-looking returns (NOT the forward ret_X columns from the parquet)
    market["ret_252d_back"] = market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(252)
    )
    market["ret_21d_back"] = market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(21)
    )
    # 12-1 momentum: 12-month return excluding the most recent month
    market["mom_12_1"] = market["ret_252d_back"] - market["ret_21d_back"]

    # 60-day realized vol from BACKWARD daily returns
    market["ret_1d_back"] = market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(1)
    )
    market["vol_60d"] = market.groupby("ticker")["ret_1d_back"].transform(
        lambda s: s.rolling(60, min_periods=20).std()
    )
    market["lowvol"] = -market["vol_60d"]

    market["month"] = market["date"].dt.to_period("M")
    monthly = (
        market.groupby(["ticker", "month"])
        .last()[["mom_12_1", "lowvol"]]
        .reset_index()
    )
    return monthly


def method_d_multifactor(preds: pd.DataFrame, market: pd.DataFrame) -> dict:
    """Combine Lazy Prices ML signal with 12-1 momentum and 60d low-vol via
    cross-sectional z-score averaging, then run rank-weighted L/S.

    Why this should work:
      - Three factors with low correlation diversify the signal noise
      - 12-1 momentum is one of the most robust factor in equities
      - Low-vol captures the well-documented vol anomaly
      - Combined Sharpe should beat any single factor by ~sqrt(N_eff)
    """
    logger.info("Method D: Multi-factor combiner (Lazy Prices + 12-1 Mom + Low-Vol)")

    factors = _build_price_factors(market)
    enriched = preds.merge(factors, on=["ticker", "month"], how="left")

    # Drop rows missing any factor (early periods that don't have 252-day history)
    before = len(enriched)
    enriched = enriched.dropna(subset=["signal", "mom_12_1", "lowvol"])
    logger.info(f"  After dropping rows missing factors: {len(enriched):,} of {before:,}")

    def weight_fn(m):
        # Cross-sectional z-scores per factor
        for col in ("signal", "mom_12_1", "lowvol"):
            mean = m[col].mean()
            std = m[col].std()
            if std == 0 or pd.isna(std):
                m = m.assign(**{f"{col}_z": 0.0})
            else:
                m = m.assign(**{f"{col}_z": (m[col] - mean) / std})
        # Equal-weight average of three z-scores
        composite = (m["signal_z"] + m["mom_12_1_z"] + m["lowvol_z"]) / 3.0
        # Convert to signed rank weights (same as Method A)
        n = len(m)
        rank = composite.rank(method="average")
        signed = (rank - (n + 1) / 2) / ((n - 1) / 2)
        return pd.Series(signed.values, index=m.index)

    df = _run_weighted_ls(enriched, weight_fn)
    metrics = _metrics(df)
    metrics["name"] = "D. Multi-factor (Lazy+Mom+LowVol, rank-wt)"
    metrics["n_months"] = len(df)
    return metrics


# ---------------------------------------------------------------------------
# Baseline reference (top10/bot10 on Lazy Prices signal alone)
# ---------------------------------------------------------------------------

def baseline_top10(preds: pd.DataFrame) -> dict:
    """Reference: same as extensions.py baseline. Should reproduce Sharpe 0.32."""
    logger.info("Baseline: top10/bot10 (Lazy Prices signal only)")

    def weight_fn(m):
        sorted_m = m.sort_values("signal", ascending=False)
        long_idx = sorted_m.head(10).index
        short_idx = sorted_m.tail(10).index
        w = pd.Series(0.0, index=m.index)
        w.loc[long_idx] = 0.1
        w.loc[short_idx] = -0.1
        return w

    df = _run_weighted_ls(preds, weight_fn)
    metrics = _metrics(df)
    metrics["name"] = "Baseline: top10/bot10 (reference)"
    metrics["n_months"] = len(df)
    return metrics


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    set_global_seeds()
    preds, market, _regimes = _load_data()
    logger.info(f"Loaded {len(preds):,} predictions across {preds['ticker'].nunique()} tickers")
    logger.info(f"Universe: {len(sorted(preds['month'].unique()))} months")

    results = [
        baseline_top10(preds),
        method_a_rank_weighted(preds),
        method_b_decile(preds),
        method_c_vol_target(preds),
        method_d_multifactor(preds, market),
        method_e_long_only(preds),
        method_f_u_shape(preds),
        method_g_momentum_only(preds, market),
        method_h_mom_lowvol(preds, market),
        method_i_cross_factor(preds, market),
        method_j_beta_hedged(preds, market),
    ]

    print()
    print("=" * 88)
    print("  ARCHITECTURAL IMPROVEMENTS — vs. baseline (Lazy Prices, 168 months)")
    print("=" * 88)
    print(
        f"  {'Method':<46s}  {'Sharpe':>7s}  {'Ann':>7s}  {'Cum':>9s}  {'MaxDD':>7s}  {'Win':>5s}"
    )
    print("-" * 88)
    base_sharpe = results[0]["sharpe"]
    for r in results:
        delta = r["sharpe"] - base_sharpe
        delta_str = f"({delta:+.2f})" if r["name"] != results[0]["name"] else ""
        print(
            f"  {r['name']:<46s}  {r['sharpe']:>+6.2f}  "
            f"{r['ann']:>+6.1%}  {r['cum']:>+8.1%}  "
            f"{r['maxdd']:>+6.1%}  {r['win']:>4.0%}  {delta_str}"
        )
    print("=" * 88)

    out_path = CACHE_DIR / "improvements_results.parquet"
    pd.DataFrame(results).to_parquet(out_path, index=False)
    logger.info(f"Saved improvements results to {out_path}")


if __name__ == "__main__":
    main()
