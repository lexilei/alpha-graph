"""Per-ticker slippage model based on Amihud illiquidity / ADV.

The existing `cost_sensitivity.py` assumes flat all-in cost per rebalance
(20 bps/month). That's optimistic for bottom-decile Lazy Prices picks, which
are often small-cap names with wide spreads and thin volume. A single flat
number overestimates alpha for illiquid books and underestimates for mega-caps.

This model:
  1. Computes trailing 60-day average dollar volume (ADV) per ticker-month.
  2. Ranks tickers cross-sectionally by ADV each month.
  3. Maps the rank to a per-ticker monthly cost via a piecewise linear curve
     calibrated to published US equity slippage estimates:
       top decile ADV:  5 bps/month round-trip
       middle:         15 bps/month
       bottom decile:  60 bps/month
     (monthly because we rebalance monthly; round-trip = buy + later sell.)
  4. Replays Method E (long top10) and Method J (long top10 − β·SPY) with
     the per-ticker cost on each long leg's picks, and reports Sharpe delta
     vs the flat 10 bps assumption.

Usage:
    python -m alpha_graph.backtest.realistic_slippage
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from alpha_graph.backtest.extensions import _load_data, _metrics
from alpha_graph.config import CACHE_DIR, set_global_seeds


# Piecewise curve: ADV rank percentile → monthly round-trip cost (bps)
# Calibrated loosely to Frazzini-Israel-Moskowitz (2018) and Engle/Ferstenberg.
COST_CURVE_BPS = [
    (1.00, 5),    # top decile ADV — SPY-like liquidity
    (0.75, 10),
    (0.50, 15),
    (0.25, 30),
    (0.10, 45),
    (0.00, 60),   # bottom decile — small-cap / thin
]


def compute_adv(market: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Trailing 60-day average dollar volume at month-end per ticker.

    Dollar volume = close × volume (volume is in shares).
    """
    m = market.sort_values(["ticker", "date"]).copy()
    m["dollar_vol"] = m["close"] * m["volume"]
    m["adv"] = m.groupby("ticker")["dollar_vol"].transform(
        lambda s: s.rolling(window, min_periods=20).mean()
    )
    m["month"] = m["date"].dt.to_period("M")
    month_end = m.groupby(["ticker", "month"]).tail(1)
    return month_end[["ticker", "month", "adv"]]


def adv_rank_to_cost_bps(rank_pct: float) -> float:
    """Piecewise linear interp on the COST_CURVE_BPS table."""
    if pd.isna(rank_pct):
        return 60.0  # treat unknown as worst case
    curve = sorted(COST_CURVE_BPS, key=lambda p: p[0])
    if rank_pct <= curve[0][0]:
        return curve[0][1]
    if rank_pct >= curve[-1][0]:
        return curve[-1][1]
    for i in range(len(curve) - 1):
        lo_p, lo_c = curve[i]
        hi_p, hi_c = curve[i + 1]
        if lo_p <= rank_pct <= hi_p:
            w = (rank_pct - lo_p) / (hi_p - lo_p)
            return lo_c + w * (hi_c - lo_c)
    return 60.0


def get_spy_monthly(start, end) -> pd.Series:
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if hasattr(spy.columns, "levels"):
        spy.columns = spy.columns.get_level_values(0)
    m = spy["Close"].resample("ME").last().pct_change().dropna()
    m.index = pd.to_datetime(m.index).to_period("M")
    return m


def run_with_realistic_costs(
    preds: pd.DataFrame,
    market: pd.DataFrame,
    hedge_spy: bool = False,
) -> pd.DataFrame:
    """Reproduce Method E (or J if hedge_spy) with per-ticker slippage."""
    adv = compute_adv(market)
    adv["rank_pct"] = adv.groupby("month")["adv"].transform(
        lambda s: s.rank(pct=True)
    )

    enriched = preds.merge(adv, on=["ticker", "month"], how="left")
    enriched["cost_bps"] = enriched["rank_pct"].apply(adv_rank_to_cost_bps)

    months = sorted(enriched["month"].unique())

    spy_series = None
    if hedge_spy:
        start = enriched["month"].min().to_timestamp().strftime("%Y-%m-%d")
        end = (enriched["month"].max().to_timestamp() + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
        spy_series = get_spy_monthly(start, end)

    # First pass: long-only returns for beta estimation (only needed if hedging)
    long_returns = {}
    for month in months:
        m = enriched[enriched["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue
        sorted_m = m.sort_values("signal", ascending=False)
        long_returns[month] = float(sorted_m.head(10)["actual_return"].mean())
    long_series = pd.Series(long_returns).sort_index()

    rows = []
    for month in months:
        m = enriched[enriched["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue
        sorted_m = m.sort_values("signal", ascending=False)
        long_pick = sorted_m.head(10)
        long_r = float(long_pick["actual_return"].mean())
        long_cost = float(long_pick["cost_bps"].mean()) / 10_000.0

        if hedge_spy:
            if spy_series is None or month not in spy_series.index:
                continue
            history = long_series.loc[:month].iloc[:-1].tail(24)
            spy_hist = spy_series.reindex(history.index).dropna()
            history = history.loc[spy_hist.index]
            if len(history) >= 12 and spy_hist.var() > 0:
                beta = float(np.cov(history.values, spy_hist.values, ddof=0)[0, 1] / spy_hist.var())
            else:
                beta = 1.0
            beta = max(0.0, min(2.0, beta))
            # SPY leg cost is the top-bucket cost (SPY is maximally liquid)
            spy_cost = adv_rank_to_cost_bps(1.0) / 10_000.0
            spy_r = float(spy_series.loc[month])
            net = long_r - beta * spy_r - long_cost - spy_cost
        else:
            beta = np.nan
            net = long_r - long_cost

        rows.append({
            "month": month.to_timestamp(),
            "net": net,
            "long_cost_bps": long_cost * 10_000,
            "beta": beta,
        })
    return pd.DataFrame(rows)


def main():
    set_global_seeds()
    preds, market, _ = _load_data()
    logger.info(f"Loaded {len(preds):,} predictions, {len(market):,} market rows")

    # Baseline flat costs (for comparison)
    FLAT_LONG = 0.001    # 10 bps/mo
    FLAT_HEDGE = 0.0015  # 15 bps/mo

    # --- Method E with realistic costs ---
    e_real = run_with_realistic_costs(preds, market, hedge_spy=False)
    e_flat = e_real.copy()
    e_flat["net"] = e_flat["net"] + (e_flat["long_cost_bps"] / 10_000.0) - FLAT_LONG

    m_real_e = _metrics(e_real)
    m_flat_e = _metrics(e_flat)

    # --- Method J with realistic costs ---
    j_real = run_with_realistic_costs(preds, market, hedge_spy=True)
    if not j_real.empty:
        j_flat = j_real.copy()
        spy_flat_cost = 0.0005
        # reconstruct "flat" cost version: add back realistic long cost + spy cost, subtract flat 15 bps
        j_flat["net"] = j_flat["net"] + (j_flat["long_cost_bps"] / 10_000.0) + spy_flat_cost - FLAT_HEDGE
        m_real_j = _metrics(j_real)
        m_flat_j = _metrics(j_flat)
    else:
        m_real_j = m_flat_j = {"sharpe": float("nan"), "ann": float("nan"), "cum": float("nan"), "maxdd": float("nan"), "win": float("nan")}

    # --- Report ---
    print()
    print("=" * 80)
    print("  REALISTIC SLIPPAGE MODEL — per-ticker ADV-ranked cost")
    print("=" * 80)
    print(f"  Cost curve (ADV rank percentile → monthly round-trip bps):")
    for p, c in sorted(COST_CURVE_BPS, key=lambda x: -x[0]):
        print(f"    {int(p*100):>3d}th pct: {c:>3d} bps/mo")
    print()
    print(f"  Avg per-month long-leg cost (Method E picks): "
          f"{e_real['long_cost_bps'].mean():.1f} bps "
          f"(vs {FLAT_LONG*10_000:.0f} bps flat assumption)")
    print(f"  Min / median / max: "
          f"{e_real['long_cost_bps'].min():.1f} / "
          f"{e_real['long_cost_bps'].median():.1f} / "
          f"{e_real['long_cost_bps'].max():.1f} bps")
    print()
    print(f"  {'Variant':<42s}  {'Sharpe':>7s}  {'Ann':>7s}  {'MaxDD':>7s}")
    print("  " + "-" * 76)
    print(f"  {'Method E — flat 10 bps':<42s}  {m_flat_e['sharpe']:>+6.2f}  {m_flat_e['ann']:>+6.1%}  {m_flat_e['maxdd']:>+6.1%}")
    print(f"  {'Method E — realistic per-ticker cost':<42s}  {m_real_e['sharpe']:>+6.2f}  {m_real_e['ann']:>+6.1%}  {m_real_e['maxdd']:>+6.1%}")
    print(f"  {'Method J — flat 15 bps':<42s}  {m_flat_j['sharpe']:>+6.2f}  {m_flat_j['ann']:>+6.1%}  {m_flat_j['maxdd']:>+6.1%}")
    print(f"  {'Method J — realistic per-ticker cost':<42s}  {m_real_j['sharpe']:>+6.2f}  {m_real_j['ann']:>+6.1%}  {m_real_j['maxdd']:>+6.1%}")
    print("=" * 80)

    # Save
    e_real.to_parquet(CACHE_DIR / "method_e_realistic_costs.parquet", index=False)
    j_real.to_parquet(CACHE_DIR / "method_j_realistic_costs.parquet", index=False)
    pd.DataFrame([
        {"variant": "E_flat", **m_flat_e},
        {"variant": "E_realistic", **m_real_e},
        {"variant": "J_flat", **m_flat_j},
        {"variant": "J_realistic", **m_real_j},
    ]).to_parquet(CACHE_DIR / "realistic_slippage_summary.parquet", index=False)
    logger.info("Saved realistic slippage outputs")


if __name__ == "__main__":
    main()
