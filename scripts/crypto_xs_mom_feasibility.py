"""Fee-net feasibility scan: cross-sectional momentum on Binance USDT perps.

Purpose: decide whether daily-rebalanced cross-sectional strategies on perps
clear realistic costs at personal size. This is a feasibility scan, not a
factor claim: the full pre-committed grid (lookback x skip) is reported, no
variant selection.

Design (fixed before results were seen):
  universe  top 100 by trailing 30d median quote volume, >=30d history,
            stablecoin pairs excluded; survivorship-free (delisted included)
  signal    momentum = close[t-skip] / close[t-skip-lb] - 1
            grid: lb in {7, 30, 90} x skip in {0, 1}
  portfolio quintile long-short, equal weight, 1.0 gross per side,
            signal at close t -> position earns close t -> close t+1
  costs     traded notional x fee, fee in {0bp gross, 2bp maker, 5bp taker}
  funding   position-weighted funding accrued (t, t+1], reported separately

Output: reports/crypto_xs_mom_feasibility.csv + printed table.

Usage: .venv/bin/python scripts/crypto_xs_mom_feasibility.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "binance"

UNIVERSE_N = 100
QUANTILE = 5
MIN_HISTORY_D = 30
STABLE_PAIRS = {"USDCUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "FDUSDUSDT",
                "AEURUSDT", "USD1USDT"}
FEE_BPS = {"gross": 0.0, "net_maker_2bp": 2e-4, "net_taker_5bp": 5e-4}
GRID = [(lb, skip) for lb in (7, 30, 90) for skip in (0, 1)]
ANN = 365


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k = pd.read_parquet(RAW / "perp_klines_1d.parquet")
    k = k[~k.symbol.isin(STABLE_PAIRS)]
    k["date"] = pd.to_datetime(k["date"])
    k = k.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    close = k.pivot(index="date", columns="symbol", values="close")
    qvol = k.pivot(index="date", columns="symbol", values="quote_volume")

    f = pd.read_parquet(RAW / "perp_funding.parquet")
    f["date"] = f["funding_time"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    fund = f.groupby(["date", "symbol"]).funding_rate.sum().unstack()
    fund = fund.reindex(index=close.index, columns=close.columns).fillna(0.0)
    return close, qvol, fund


def build_universe(close: pd.DataFrame, qvol: pd.DataFrame) -> pd.DataFrame:
    """Boolean mask: tradable member of the top-N liquidity universe at t."""
    med_qv = qvol.rolling(MIN_HISTORY_D, min_periods=MIN_HISTORY_D).median()
    has_price = close.notna()
    rank = med_qv.where(has_price).rank(axis=1, ascending=False)
    return (rank <= UNIVERSE_N) & has_price


def run_variant(close: pd.DataFrame, universe: pd.DataFrame,
                fund: pd.DataFrame, lb: int, skip: int) -> dict:
    ret = close.pct_change(fill_method=None)
    mom = close.shift(skip) / close.shift(skip + lb) - 1
    sig = mom.where(universe)

    r = sig.rank(axis=1)
    n = r.max(axis=1)
    top = r.ge(n * (1 - 1 / QUANTILE), axis=0)
    bot = r.le(n / QUANTILE, axis=0)
    w = top.div(top.sum(axis=1), axis=0).fillna(0.0) \
        - bot.div(bot.sum(axis=1), axis=0).fillna(0.0)
    w[n < QUANTILE * 4] = 0.0  # need enough names to form quintiles

    # position set at close t earns t -> t+1; funding accrues on held position
    r_fwd = ret.shift(-1)
    pnl_price = (w * r_fwd).sum(axis=1)
    pnl_fund = -(w * fund.shift(-1)).sum(axis=1)

    # traded notional per unit NAV vs drifted weights; delisting closeout paid
    w_drift = (w.shift(1) * (1 + ret)).fillna(0.0)
    traded = (w - w_drift).abs().sum(axis=1)

    live = w.abs().sum(axis=1) > 0
    out = {"lb": lb, "skip": skip,
           "start": str(pnl_price[live].index.min().date()),
           "days": int(live.sum()),
           "turnover_1s": float(traded[live].mean() / 2)}
    for fee_name, fee in FEE_BPS.items():
        net = (pnl_price + pnl_fund - traded * fee)[live]
        eq = (1 + net).cumprod()
        dd = float((eq / eq.cummax() - 1).min())
        sr = float(net.mean() / net.std() * np.sqrt(ANN)) if net.std() > 0 else np.nan
        out[f"{fee_name}_ann"] = float(net.mean() * ANN)
        out[f"{fee_name}_sharpe"] = sr
        if fee_name == "net_taker_5bp":
            out["maxdd_taker"] = dd
    net23 = (pnl_price + pnl_fund - traded * FEE_BPS["net_taker_5bp"])[live]
    net23 = net23[net23.index >= "2023-01-01"]
    out["taker_sharpe_2023on"] = (
        float(net23.mean() / net23.std() * np.sqrt(ANN)) if len(net23) > 100 else np.nan)
    out["fund_ann"] = float(pnl_fund[live].mean() * ANN)
    return out


def main() -> None:
    close, qvol, fund = load_panels()
    universe = build_universe(close, qvol)
    print(f"panel: {close.shape[0]} days x {close.shape[1]} symbols, "
          f"{close.index.min().date()} .. {close.index.max().date()}")
    print(f"universe: top {UNIVERSE_N} by 30d median quote volume; "
          f"mean members/day = {universe.sum(axis=1).mean():.0f}")

    rows = [run_variant(close, universe, fund, lb, skip) for lb, skip in GRID]
    res = pd.DataFrame(rows)
    out = ROOT / "reports" / "crypto_xs_mom_feasibility.csv"
    res.to_csv(out, index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(res.round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
