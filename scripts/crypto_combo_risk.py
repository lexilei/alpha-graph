"""Portfolio-construction diagnostics for the 3-sleeve crypto combo.

Not a factor look: sleeves and weights are fixed (mom30s1 + K2_carry_7d +
K11_amihud_30d, equal risk, from crypto_factor_scan_k1.md). Questions:
  1. Does the combo carry residual BTC beta, and does an implementable
     ex-ante hedge (position-weighted 60d betas, BTC overlay at taker fee)
     change its Sharpe?
  2. What does 15%-vol targeting (trailing 30d vol, leverage capped at 3x)
     deliver in return / drawdown terms?

Output: printed table + reports/crypto_combo_risk.csv.

Usage: .venv/bin/python scripts/crypto_combo_risk.py
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
SLEEVES = ("mom30s1", "K2_carry_7d", "K11_amihud_30d")
VOL_TARGET = 0.15
LEV_CAP = 3.0


def sleeve_weights(sig, universe):
    s = sig.where(universe)
    r = s.rank(axis=1)
    n = r.max(axis=1)
    top = r.ge(n * (1 - 1 / scan.QUANTILE), axis=0)
    bot = r.le(n / scan.QUANTILE, axis=0)
    w = top.div(top.sum(axis=1), axis=0).fillna(0.0) \
        - bot.div(bot.sum(axis=1), axis=0).fillna(0.0)
    w[n < scan.QUANTILE * 4] = 0.0
    return w


def stats(x: pd.Series, label: str) -> dict:
    eq = (1 + x).cumprod()
    m = x.resample("ME").sum()
    return {"variant": label,
            "ann_ret": float(x.mean() * ANN),
            "ann_vol": float(x.std() * np.sqrt(ANN)),
            "sharpe": float(x.mean() / x.std() * np.sqrt(ANN)),
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "worst_month": float(m.min()),
            "sharpe_2023on": float(
                x[x.index >= "2023-01-01"].mean()
                / x[x.index >= "2023-01-01"].std() * np.sqrt(ANN))}


def main() -> None:
    p = scan.load()
    close, qv, fund = p["close"], p["quote_volume"], p["fund"]
    ret = close.pct_change(fill_method=None)
    med = qv.rolling(30, min_periods=30).median()
    universe = (med.where(close.notna())
                .rank(axis=1, ascending=False) <= 100) & close.notna()
    sigs = scan.build_signals(p)

    btc = ret["BTCUSDT"]
    beta = ret.rolling(60).cov(btc).div(btc.rolling(60).var(), axis=0)

    nets, exante_betas = {}, {}
    for name in SLEEVES:
        w = sleeve_weights(sigs[name], universe)
        pnl = (w * ret.shift(-1)).sum(axis=1) - (w * fund.shift(-1)).sum(axis=1)
        traded = (w - (w.shift(1) * (1 + ret)).fillna(0.0)).abs().sum(axis=1)
        live = w.abs().sum(axis=1) > 0
        nets[name] = (pnl - traded * TAKER)[live]
        exante_betas[name] = (w * beta).sum(axis=1)[live]

    net = pd.DataFrame(nets).dropna()
    eb = pd.DataFrame(exante_betas).reindex(net.index)
    inv_vol = 1.0 / net.std()
    inv_vol /= inv_vol.sum()
    combo = (net * inv_vol).sum(axis=1)
    combo_beta = (eb * inv_vol).sum(axis=1)

    # realized beta check (regression of combo on next-day BTC alignment)
    btc_fwd = btc.shift(-1).reindex(combo.index)
    realized_beta = float(combo.cov(btc_fwd) / btc_fwd.var())

    # implementable overlay: short combo_beta units of BTC, taker cost on changes
    hedge_pnl = -combo_beta * btc_fwd - combo_beta.diff().abs().fillna(0) * TAKER
    hedged = combo + hedge_pnl

    # vol targeting on the hedged series; leverage change costs scale with
    # traded notional ~ |dL| * 2.0 gross at taker
    lev = (VOL_TARGET / (hedged.rolling(30).std() * np.sqrt(ANN))).clip(upper=LEV_CAP)
    lev = lev.shift(1)
    vt = (hedged * lev - lev.diff().abs().fillna(0) * 2.0 * TAKER).dropna()

    print(f"combo ex-ante BTC beta: mean {combo_beta.mean():+.3f}, "
          f"5%..95% [{combo_beta.quantile(0.05):+.3f}, "
          f"{combo_beta.quantile(0.95):+.3f}] | realized {realized_beta:+.3f}")
    rows = [stats(combo, "raw_equal_risk"),
            stats(hedged, "btc_hedged"),
            stats(vt, f"hedged_voltarget_{int(VOL_TARGET*100)}pct")]
    res = pd.DataFrame(rows)
    out = ROOT / "reports" / "crypto_combo_risk.csv"
    res.to_csv(out, index=False, float_format="%.4f")
    print(res.round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
