#!/usr/bin/env python3
"""Validate alpha_graph.options.iv against vol_smile's audited SPY ATM-IV
series (reports/atm_iv_daily_spy.csv, engine v2.1).

Same method (37±9 DTE nearest expiry, nearest strike, both-leg bisect);
TWO intended differences: (1) spot — parity-implied FORWARD here vs
spot.parquet close there; (2) inversion — Black-76 against that forward
here (2026-07-17 audit fix; kills the −3.2 vol-pt leg wedge) vs spot-BS
at rate=0.04 there. On SPY (no splits) the leg-AVERAGED ATM IV must still
agree closely — this run validates the verbatim BS port, the parity-spot
construction, and the Black-76 change before they touch single names
(post-fix: same-selection median 8bp, leg wedge median 0.0 ± 0.6bp).

Pass gate (printed verdict, REVISED after the first run's decomposition —
the original single-number gate conflated selection-convention differences
with numerical error):
- coverage >= 95% of audited days;
- SAME-SELECTION days (same expiry AND strike as the audited series):
  n >= 500 and median |ΔIV| <= 0.001 — this is the numerical-correctness
  proof (first run: n=710, median 0.00038);
- overall median |ΔIV| <= 0.005 and p95 <= 0.015 — bounds the
  convention-difference classes (parity/forward spot picks a strike 1-2
  grid steps from the close-spot pick ~57% of days; adjacent-strike IV
  differs by the real smile slope, ~30bp — a design choice, not error);
- median |Δspot|/spot <= 0.3% (parity spot ≈ forward: S·(1+(r−q)T),
  ~+0.23% at 37 DTE — expected and documented).

Data engineering, not a look.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alpha_graph.options.iv import atm_iv_for_day  # noqa: E402

CLEAN = "/Users/leizhihan/Lex/data/vol_smile/clean"
AUDITED = "/Users/leizhihan/Lex/vol_smile/reports/atm_iv_daily_spy.csv"
OUT = Path(__file__).resolve().parents[1] / "reports" / "iv_pipeline_validation_spy.csv"

COLS = ["quote_date", "expiration", "strike", "right", "bid", "ask", "created"]


def main() -> int:
    audited = pd.read_csv(AUDITED).set_index("qd")
    files = sorted(glob.glob(f"{CLEAN}/year=*/month=*/quote_date=*/data.parquet"))
    print(f"{len(files)} day files, {len(audited)} audited days")

    rows = []
    for f in files:
        qd = f.split("quote_date=")[1].split("/")[0]
        day = pd.read_parquet(f, columns=COLS)
        r = atm_iv_for_day(day, qd)
        if r is None or r.iv is None:
            rows.append({"qd": qd, "iv_ours": None, "spot_impl": None})
            continue
        rows.append(
            {"qd": qd, "iv_ours": r.iv, "spot_impl": r.spot_impl,
             "dte_ours": r.dte, "strike_ours": r.strike,
             "iv_call": r.iv_call, "iv_put": r.iv_put}
        )
    ours = pd.DataFrame(rows).set_index("qd")
    j = audited.join(ours, how="left")
    j["d_iv"] = (j["iv_ours"] - j["atm_iv"]).abs()
    j["d_spot"] = (j["spot_impl"] - j["spot"]).abs() / j["spot"]
    OUT.parent.mkdir(exist_ok=True)
    j.to_csv(OUT)

    have = j["iv_ours"].notna()
    cov = have.mean()
    med_iv, p95_iv = j["d_iv"].median(), j["d_iv"].quantile(0.95)
    med_spot = j["d_spot"].median()
    same_e = j["dte_ours"] == j["dte"]
    same_sel = same_e & (j["strike_ours"] == j["strike"])
    sel = j.loc[same_sel, "d_iv"]
    print(f"reproduced days: {have.sum()}/{len(j)} ({cov:.1%})")
    print(f"|ΔIV|  median {med_iv:.5f}  p95 {p95_iv:.5f}  max {j['d_iv'].max():.5f}")
    print(f"|Δspot|/spot median {med_spot:.5%}")
    print(f"same expiry picked: {same_e.mean():.1%}  same exp+strike: {same_sel.mean():.1%}")
    print(f"same-selection |ΔIV|: n={len(sel)} median {sel.median():.5f}")
    # Leg-level diagnostic (2026-07-17 audit, finding 2: the leg AVERAGE is
    # the one combination where a carry-double-count wedge cancels — report
    # the call-put spread so a leg-level defect can't hide behind the gate).
    # The audited CSV has no leg columns, so this is a self-diagnostic, not
    # a comparison: post-Black-76 the same-strike spread should be small
    # (residual = American early-exercise premium + quote noise, not carry).
    leg_spread = j["iv_call"] - j["iv_put"]
    print(
        f"iv_call - iv_put: median {leg_spread.median():+.5f}  "
        f"p5 {leg_spread.quantile(0.05):+.5f}  p95 {leg_spread.quantile(0.95):+.5f}"
    )
    ok = (
        cov >= 0.95
        and len(sel) >= 500 and sel.median() <= 0.001
        and med_iv <= 0.005 and p95_iv <= 0.015
        and med_spot <= 0.003
    )
    print("VERDICT:", "PASS" if ok else "FAIL")
    print(f"wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
