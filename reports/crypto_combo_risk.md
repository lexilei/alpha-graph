# Combo risk diagnostics: beta hedge and vol targeting

2026-07-21. Construction-level diagnostics on the fixed 3-sleeve combo
(mom30s1 + K2 + K11, equal risk; see `crypto_factor_scan_k1.md`). Not
factor looks; one configuration each, no tuning. Numbers:
`crypto_combo_risk.csv`, script `scripts/crypto_combo_risk.py`.

## Results

| variant | ann ret | ann vol | SR | maxDD | worst mo | SR 2023+ |
|---|---:|---:|---:|---:|---:|---:|
| raw equal-risk | 45.7% | 28.7% | **1.59** | −41% | −12.9% | **1.20** |
| + ex-ante BTC hedge | 45.2% | 29.5% | 1.53 | −44% | −10.6% | 0.93 |
| + 15% vol target (30d, cap 3×) | 22.3% | 17.2% | 1.30 | −33% | −9.1% | 0.81 |

Ex-ante BTC beta of the raw combo: mean +0.000, 5–95% range
[−0.25, +0.21], realized full-sample beta −0.03.

## Reading — both overlays rejected

1. **The combo is already BTC-neutral on average.** Quintile EW long-short
   nets out beta by construction; only time-varying wobble remains.
   Actively hedging that wobble with a BTC overlay trades noisy beta
   estimates at taker fees and *lowers* Sharpe (1.59 → 1.53 full,
   1.20 → 0.93 recent). Rejected; keep beta as a monitored diagnostic.
2. **Naive vol targeting hurts per-unit-risk performance here** (SR
   1.53 → 1.30, and vol-normalized maxDD worsens: 1.43σ-units → 1.91).
   Mechanism: the sleeves earn disproportionately in high-vol regimes, so
   trailing-vol deleveraging clips the best days. The absolute drawdown
   "improvement" is just lower vol, obtainable more cheaply by static
   scaling. Rejected in this form. Alternatives (expected-vol from
   positions, shorter windows) were NOT tried — trying several and keeping
   the best would be construction overfitting; if revisited, pre-commit.
3. **Practical conclusion:** run the raw combo at *static* leverage chosen
   for the desired vol. E.g. ~0.52× gross → ~15% vol, backtest ann ret
   ≈ 24%, maxDD ≈ −21%, worst month ≈ −6.7%, Sharpe unchanged 1.59/1.20.
   Live expectation stays as stated previously: shade Sharpe to 0.8–1.2.
