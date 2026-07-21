#!/usr/bin/env python3
"""Empirical null calibration of the v0 judge (infrastructure, NOT looks).

Runs the exact production judge (load_panel → to_monthly → evaluate,
sn incremental over BASELINE) on BATCHES of synthetic signals with KNOWN
zero information, generated directly on the panel's (ticker, month)
grid. Purpose: measure the judge's actual false-positive geometry —

  - is the naive t ~N(0,1) under the null, per persistence regime?
  - what do the 1.5 bar and the emax_null ceiling correspond to
    EMPIRICALLY for slow signals? (The 2026-07-20 integration audit
    showed naive t overstates for high-persistence candidates — C30's
    −3.12 read −2.4..−3.0 under HAC.)

Regimes (per-ticker AR(1) in monthly innovations, cross-sectionally
independent): iid (rho=0), fast (rho=0.5), slow (rho=0.98 — the
volume_trend / fundamentals class, monthly rank-autocorr ≈ 0.99).

These runs are NOT hypothesis tests about markets — a random signal
cannot be selected or traded — so they do not enter selection-N; the
run is recorded in the ledger as infrastructure with this rationale.

Usage: .venv/bin/python scripts/judge_null_calibration.py [K_per_regime]
Output: reports/judge_null_calibration.csv + stdout summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from factor_orthogonality import BASELINE, evaluate, load_panel, to_monthly  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "judge_null_calibration.csv"
REGIMES = {"iid": 0.0, "fast": 0.5, "slow": 0.98}


def make_null_signal(panel_m: pd.DataFrame, rho: float,
                     rng: np.random.Generator) -> pd.Series:
    """Per-ticker AR(1) over that ticker's month sequence."""
    out = np.empty(len(panel_m))
    scale = np.sqrt(1 - rho**2) if rho < 1 else 1.0
    for _, idx in panel_m.groupby("ticker").indices.items():
        n = len(idx)
        eps = rng.standard_normal(n)
        s = np.empty(n)
        s[0] = eps[0]
        for i in range(1, n):
            s[i] = rho * s[i - 1] + scale * eps[i]
        out[idx] = s
    return pd.Series(out, index=panel_m.index)


def main() -> int:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    panel = load_panel(None, pit=True, lag=1, daily=True)
    panel_m = to_monthly(panel).reset_index(drop=True)
    panel_m = panel_m.sort_values(["ticker", "month"]).reset_index(drop=True)
    rng = np.random.default_rng(20260721)
    rows = []
    for regime, rho in REGIMES.items():
        for j in range(k):
            col = f"null_{regime}_{j}"
            panel_m[col] = make_null_signal(panel_m, rho, rng)
            res = evaluate(panel_m, BASELINE, col, sector_neutral=True)
            panel_m = panel_m.drop(columns=col)
            if res.get("ok"):
                rows.append(
                    {"regime": regime, "j": j,
                     "t_incr": res["ic_incremental_t"],
                     "t_raw": res["ic_raw_t"],
                     "n_months": res["n_months"]}
                )
        done = pd.DataFrame(rows)
        sub = done[done["regime"] == regime]["t_incr"]
        print(f"{regime} (rho={rho}): n={len(sub)}  std(t)={sub.std(ddof=1):.3f}  "
              f"P(|t|>=1.5)={(sub.abs() >= 1.5).mean():.3f}  "
              f"P(|t|>=2.92)={(sub.abs() >= 2.92).mean():.3f}  "
              f"max|t|={sub.abs().max():.2f}")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
