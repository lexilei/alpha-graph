"""Rerun the 4-sleeve crypto combo on the US-legal (Coinbase perp) universe.

The full-universe result (mom30s1 + K2 + K11 + K12, net-taker SR 2.07/1.75)
was built on all Binance USDT perps — a venue the user cannot legally trade.
This rerun restricts the universe to symbols whose base asset has a Coinbase
perpetual (fetched live from the public products API), as the executable
approximation. Caveats printed with the result: Coinbase US-retail eligibility
is a subset of the INTX list (verify after account opening); prices/funding
are still Binance proxies; Coinbase fee tiers differ from the 5bp taker used.

Usage: .venv/bin/python scripts/crypto_restricted_universe.py
"""

from __future__ import annotations

import importlib.util
import json
import urllib.request
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
SLEEVES = ("mom30s1", "K2_carry_7d", "K11_amihud_30d", "K12_tbr_7d")


def coinbase_perp_bases() -> set[str]:
    url = ("https://api.coinbase.com/api/v3/brokerage/market/products"
           "?product_type=FUTURE&contract_expiry_type=PERPETUAL")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r)
    return {p["product_id"].replace("-PERP-INTX", "")
            for p in d.get("products", [])}


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(ANN)) if x.std() > 0 else np.nan


def run(universe, p, label: str) -> pd.Series:
    close, qv, fund = p["close"], p["quote_volume"], p["fund"]
    ret = close.pct_change(fill_method=None)
    sigs = scan.build_signals(p)
    nets = {}
    for s in SLEEVES:
        pnl, traded, live = scan.ls_returns(sigs[s], universe, ret, fund)
        nets[s] = (pnl - traded * TAKER)[live]
    d = pd.concat(nets, axis=1).dropna()
    combo = (d / d.std()).mean(axis=1)
    c23 = combo[combo.index >= "2023-01-01"]
    print(f"\n{label}: mean members/day = {universe.sum(axis=1).mean():.0f}")
    for s in SLEEVES:
        x = nets[s]
        x23 = x[x.index >= "2023-01-01"]
        print(f"  {s:16s} SR {sharpe(x):5.2f} | 2023+ {sharpe(x23):5.2f}")
    print(f"  {'COMBO':16s} SR {sharpe(combo):5.2f} | 2023+ {sharpe(c23):5.2f}")
    return combo


def main() -> None:
    bases = coinbase_perp_bases()
    p = scan.load()
    close, qv = p["close"], p["quote_volume"]
    panel_syms = set(close.columns)
    eligible = {b + "USDT" for b in bases} & panel_syms
    print(f"coinbase perp bases: {len(bases)} | mapped to Binance panel: "
          f"{len(eligible)}")

    med = qv.rolling(30, min_periods=30).median()
    rank_all = med.where(close.notna()).rank(axis=1, ascending=False)
    uni_full = (rank_all <= 100) & close.notna()

    elig_cols = close.columns.isin(eligible)
    med_r = med.where(close.notna()).loc[:, elig_cols]
    rank_r = med_r.rank(axis=1, ascending=False)
    uni_r = pd.DataFrame(False, index=close.index, columns=close.columns)
    uni_r.loc[:, elig_cols] = (rank_r <= 100) & close.loc[:, elig_cols].notna()

    c_full = run(uni_full, p, "FULL universe (Binance, not executable)")
    c_r = run(uni_r, p, "RESTRICTED universe (Coinbase-perp bases)")
    joint = pd.concat({"f": c_full, "r": c_r}, axis=1).dropna()
    print(f"\ncombo corr full vs restricted: {joint.f.corr(joint.r):.3f}")
    print("caveats: INTX list is the upper bound of US-retail eligibility; "
          "Binance prices/funding as proxy; Coinbase fees differ from 5bp.")


if __name__ == "__main__":
    main()
