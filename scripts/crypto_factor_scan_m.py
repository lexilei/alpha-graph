"""Evaluate pre-registered crypto factor batch M1-M5 (OI / long-short ratios).

Definitions, signs and bars frozen in reports/crypto_factor_prereg_m.md,
committed before the metrics dataset landed. Protocol identical to batch K.
Baseline for incrementality: the 3-sleeve combo (mom30s1 + K2 + K11), as
pinned in the prereg (written before K12 was judged); corr vs K12 reported
informationally.

Output: reports/crypto_factor_scan_m.csv + printed table.

Usage: .venv/bin/python scripts/crypto_factor_scan_m.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "scan", ROOT / "scripts" / "crypto_factor_scan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

TAKER = 5e-4
ANN = 365


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(ANN)) if x.std() > 0 else np.nan


def main() -> None:
    p = scan.load()
    close, qv, fund = p["close"], p["quote_volume"], p["fund"]
    ret = close.pct_change(fill_method=None)
    med = qv.rolling(30, min_periods=30).median()
    universe = (med.where(close.notna())
                .rank(axis=1, ascending=False) <= 100) & close.notna()

    m = pd.read_parquet(ROOT / "data/raw/binance/perp_metrics_daily.parquet")
    # a daily zip's snapshots can spill past midnight -> duplicate (symbol, date)
    m = m.sort_values(["symbol", "date"]).drop_duplicates(
        ["symbol", "date"], keep="last")
    pv = {c: m.pivot(index="date", columns="symbol", values=c)
          .reindex(index=close.index, columns=close.columns)
          for c in ("sum_open_interest_value_last",
                    "count_long_short_ratio_last",
                    "sum_toptrader_long_short_ratio_last")}
    oi = pv["sum_open_interest_value_last"]
    log_oi = np.log(oi.where(oi > 0))
    sigs_m = {
        "M1_oi_chg_7d": log_oi - log_oi.shift(7),
        "M2_oi_crowd": -(oi / med),
        "M3_lsr_retail_7d": -pv["count_long_short_ratio_last"].rolling(7).mean(),
        "M4_lsr_top_7d": pv["sum_toptrader_long_short_ratio_last"].rolling(7).mean(),
        "M5_oi_chg_1d": log_oi - log_oi.shift(1),
    }

    sigs_k = scan.build_signals(p)
    base = {}
    for s in ("mom30s1", "K2_carry_7d", "K11_amihud_30d", "K12_tbr_7d"):
        pnl, traded, live = scan.ls_returns(sigs_k[s], universe, ret, fund)
        base[s] = (pnl - traded * TAKER)[live]
    b3 = pd.concat({k: v for k, v in base.items() if k != "K12_tbr_7d"},
                   axis=1).dropna()
    combo3 = (b3 / b3.std()).mean(axis=1)
    combo3_sr = sharpe(combo3)

    rows = []
    for name, sig in sigs_m.items():
        pnl, traded, live = scan.ls_returns(sig, universe, ret, fund)
        net = (pnl - traded * TAKER)[live]
        net23 = net[net.index >= "2023-01-01"]
        corr = float(net.corr(combo3))
        cboth = pd.concat([net / net.std(), combo3 / combo3.std()],
                          axis=1).dropna()
        combo_new = cboth.mean(axis=1)
        sr = sharpe(net)
        rows.append({
            "id": name,
            "start": str(net.index.min().date()),
            "turnover_1s": float(traded[live].mean() / 2),
            "gross_sr": sharpe(pnl[live]),
            "taker_sr": sr,
            "t": sr * np.sqrt(len(net) / ANN) if np.isfinite(sr) else np.nan,
            "taker_sr_2023on": sharpe(net23),
            "corr_combo3": corr,
            "combo_dSR": sharpe(combo_new) - combo3_sr,
            "corr_K12": float(net.corr(base["K12_tbr_7d"])),
        })
    res = pd.DataFrame(rows)

    def verdict(r: pd.Series) -> str:
        if r.taker_sr < 0:
            return "REJECT wrong-sign"
        standalone = (r.taker_sr >= 0.5 and r.t >= 2.0 and r.taker_sr_2023on > 0)
        divers = (abs(r.corr_combo3) <= 0.4 and r.combo_dSR >= 0.10
                  and r.taker_sr >= 0.3)
        if standalone and divers:
            return "PASS standalone+div"
        if standalone:
            return "PASS standalone"
        if divers:
            return "PASS diversifier"
        return "reject"

    res["verdict"] = res.apply(verdict, axis=1)
    out = ROOT / "reports" / "crypto_factor_scan_m.csv"
    res.to_csv(out, index=False, float_format="%.4f")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(res.round(3).to_string(index=False))
    print(f"baseline 3-sleeve combo net-taker SR = {combo3_sr:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
