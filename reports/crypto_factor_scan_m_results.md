# Crypto factor batch M — results

2026-07-21 late. Batch frozen in `crypto_factor_prereg_m.md` before the
metrics dataset landed. Data: 672 symbols' OI + long-short ratios,
2020-09..2026-07 (dense from 2021-12), deduped on (symbol, date).
Numbers: `crypto_factor_scan_m.csv`. One evaluation bug was fixed mid-run:
the first combo-ΔSR compared a 2021-12+ subsample combo against the
full-sample baseline (period contamination); verdicts were issued only
after the joint-subsample fix. Verdict bars unchanged from prereg.

## Verdicts (5 looks; venue ledger now 25)

| ID | taker SR | t | 2023+ | corr combo3 | ΔSR | corr K12 | verdict |
|----|---:|---:|---:|---:|---:|---:|---|
| M3 lsr_retail_7d (fade retail net-long) | 1.02 | 2.15 | 1.15 | 0.49 | +0.28 | **0.61** | PASS standalone |
| M1 oi_chg_7d | 0.52 | 1.12 | 1.05 | 0.22 | +0.08 | 0.34 | reject (below both bars) |
| M2 oi_crowd | −0.50 | | | | | | REJECT wrong-sign |
| M4 lsr_top_7d (follow top traders) | **−1.50** | −2.84 | | | | | REJECT wrong-sign |
| M5 oi_chg_1d | −0.83 | | | | | | REJECT wrong-sign |

## Reading

1. **The metrics dataset adds almost nothing to what klines already
   carry.** M3 passes its pinned standalone bar, but it is 0.61-correlated
   with K12 (taker buy share): retail accounts being net-long and taker
   flow being net-sell are close to the same fact measured twice. Post-hoc
   diagnostic: adding M3 to the 4-sleeve combo moves joint-sample SR
   1.24 → 1.27 and 2023+ 1.77 → 1.74 — no marginal value. **M3 is
   registered as passed but does not enter the combo.**
2. **Top-trader positioning is a strong wrong-way signal** (following
   them: SR −1.50, t −2.84). A faded version would be a new registered
   look and overlaps the K12/M3 information cluster; logged as an idea,
   not spent.
3. OI level/changes carry little after costs (M1 below bar, M5 killed by
   1.55/day turnover, crowding penalty nonexistent — consistent with the
   K-batch risk-on structure).
4. Combo stands unchanged: **mom30s1 + K2 + K11 + K12, net-taker SR 2.07
   full / 1.75 2023+.**

Venue look ledger after batch M: 25 looks, 5 passes (K2, K3, K11, K12,
K14, M3 = 6 pass verdicts; 2 of them redundant), family E[max] ≈ 2.4σ;
K12 (4.85) remains the only result decisively above every ceiling.
