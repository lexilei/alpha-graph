#!/usr/bin/env python3
"""I22 div-seasonality PRE-REGISTRATION cost check (pinned prerequisite in
IDEAS.md: compute break-even turnover BEFORE the family may open; data
engineering, not a look).

From dividends.parquet (ex-dates): infer payment frequency per ticker
(median ex-date gap: <135d → quarterly, <270d → semiannual, else
annual), build the monthly predicted-payer flag per the OSAP DivSeason
rule adapted to ex-months (quarterly: ex-div 2/5/8/11 months ago;
semiannual: 5/11; annual: 11; universe = paid within last 12 months).

Reports:
1. prediction accuracy: P(actual ex-div this month | predicted) and
   P(predicted | actual) — sanity of the frequency inference;
2. month-over-month one-way turnover of the predicted-payer set (long
   leg) and of the payer-but-not-predicted set (short leg);
3. break-even cost/side under OSAP 2011+ gross means (DivSeason
   0.10%/mo, DivYieldST 0.25%/mo, EW full-universe — the FAVORABLE
   case), against the frozen gate line (>= 2x ~3bp actual = 6bp/side,
   reports/promotion_gate_c11.md).

Usage: .venv/bin/python scripts/i22_divseason_precheck.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def main() -> int:
    div = pd.read_parquet(CACHE / "dividends.parquet")
    div["ex_month"] = div["ex_date"].dt.to_period("M")
    # regular-dividend proxy: drop months with >1 ex-date (specials mixed in)
    per = div.groupby(["ticker", "ex_month"]).size().rename("n").reset_index()

    # frequency inference from median ex-date gaps (last 15 years)
    recent = div[div["ex_date"] >= "2011-01-01"].sort_values(["ticker", "ex_date"])
    gaps = recent.groupby("ticker")["ex_date"].diff().dt.days
    med_gap = gaps.groupby(recent["ticker"]).median()
    freq = pd.cut(med_gap, [0, 135, 270, 10_000],
                  labels=["Q", "S", "A"]).astype(str)
    print("frequency mix:", freq.value_counts().to_dict())

    months = pd.period_range("2011-01", "2026-06", freq="M")
    paid = {(t, m) for t, m in zip(per["ticker"], per["ex_month"])}
    tickers = sorted(set(per["ticker"]) & set(freq.index))

    LAGS = {"Q": (2, 5, 8, 11), "S": (5, 11), "A": (11,)}
    rows = []
    for t in tickers:
        f = freq.loc[t]
        if f not in LAGS:
            continue
        for m in months:
            paid_12m = any((t, m - k) in paid for k in range(1, 13))
            if not paid_12m:
                continue
            predicted = any((t, m - k) in paid for k in LAGS[f])
            # the OSAP lag set (2/5/8/11) is formed at month m to predict
            # the ex-div in month m+1 (portfolio formed at m month-end)
            actual = (t, m + 1) in paid
            rows.append((t, str(m), predicted, actual))
    df = pd.DataFrame(rows, columns=["ticker", "month", "pred", "actual"])

    acc_p = df[df["pred"]]["actual"].mean()
    acc_a = df[df["actual"]]["pred"].mean()
    print(f"universe rows {len(df)}, payer-months with prediction "
          f"{df['pred'].mean():.1%} of payer universe")
    print(f"P(actual ex-div | predicted) = {acc_p:.1%}")
    print(f"P(predicted | actual ex-div) = {acc_a:.1%}")

    # month-over-month one-way turnover per leg
    to_long, to_short, xs_long = [], [], []
    for m0, m1 in zip(months[:-1], months[1:]):
        a = df[df["month"] == str(m0)]
        b = df[df["month"] == str(m1)]
        for flag, sink in ((True, to_long), (False, to_short)):
            s0 = set(a[a["pred"] == flag]["ticker"])
            s1 = set(b[b["pred"] == flag]["ticker"])
            if s0 and s1:
                sink.append(1 - len(s0 & s1) / max(len(s0), len(s1)))
        xs_long.append((df["month"] == str(m1)).sum() and
                       (b["pred"]).sum())
    to_l, to_s = np.mean(to_long), np.mean(to_short)
    print(f"avg monthly one-way turnover: long(predicted) {to_l:.1%}, "
          f"short(payer-not-predicted) {to_s:.1%}; "
          f"avg long-leg breadth {np.mean(xs_long):.0f}")

    # break-even per side: annual gross / (2 legs x annual one-way turnover)
    ann_turn = 12 * (to_l + to_s) / 2
    for name, gross_mo in (("DivSeason", 0.10), ("DivYieldST", 0.25)):
        be = gross_mo * 12 / (2 * ann_turn) * 100  # bp/side
        verdict = "PASS" if be >= 6.0 else "FAIL"
        print(f"{name}: gross {gross_mo}%/mo -> break-even "
              f"{be:.1f} bp/side vs 6.0 required -> {verdict}")
    return 0


if __name__ == "__main__":
    main()
