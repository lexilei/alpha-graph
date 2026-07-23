# Crypto factor batch K1 — results

2026-07-21. Evaluation of the batch frozen in `crypto_factor_prereg.md`
(committed before this run). K12 was judged hours after the other 13 (the
first klines download dropped `taker_buy_volume`; re-downloaded), same
pinned bars, no definition change. Numbers: `crypto_factor_scan_k1.csv`.
Baseline mom30s1 net-taker SR 0.946 / 6.3y.

## Verdicts

| ID | net-taker SR | t | 2023+ | corr mom | combo ΔSR | verdict |
|----|---:|---:|---:|---:|---:|---|
| K12 tbr_7d | **1.93** | **4.85** | 1.93 | 0.38 | **+0.79** | PASS standalone+diversifier |
| K2 carry_7d | **1.05** | **2.63** | 0.58 | 0.04 | **+0.45** | PASS standalone+diversifier |
| K14 high_90d | 0.93 | 2.30 | 0.88 | 0.56 | +0.18 | PASS standalone (momentum cousin) |
| K11 amihud_30d | 0.68 | 1.71 | 0.34 | −0.00 | +0.21 | PASS diversifier |
| K3 carry_30d | 0.65 | 1.64 | 0.62 | 0.09 | +0.15 | PASS diversifier (corr 0.55 w/ K2) |
| K1 carry_1d | 0.19 | — | −0.06 | | | reject: turnover 0.87/d eats it |
| K10 size_qv | 0.47 | 1.18 | −0.42 | | | reject: dead 2023+ |
| K4 rev_1d, K5 rev_3d | −0.90, −1.06 | | | | | REJECT wrong sign |
| K6 lowvol, K7 bab, K8 max, K9 skew, K13 volz | −0.32 … −1.22 | | | | | REJECT wrong sign |

Family context: 14 looks this batch, venue ledger now 20 looks.
Expected-max |t| under the null for 20 looks ≈ 2.3σ. K12 (4.85) clears it
decisively — the strongest single result on any venue in this ledger; K2
(2.63) is above it; K14 (2.30) sits at it.

*(Correction note 2026-07-23: the scan's combo_dSR compared a joint-window
combo against the full-sample baseline SR — a subsample-mismatch defect found
in code review. Recomputed with the corrected formula on the original
judgment window (panel ≤ 2026-06-30): every dSR and every verdict in this
file's table is unchanged to 3 decimals — the mismatch term was nil on this
window because all pass-tier factors share the baseline's live window. The
code fix is in `crypto_factor_scan.py`; future evaluations use it.)*

## K12 verification (checked before acceptance)

- Data sanity: 7 of 587,609 kline rows have taker_buy > volume (known
  Binance glitches, all 2023); irrelevant to a 7d-mean rank signal.
- Leg decomposition: both legs work — long-leg minus universe SR +1.87,
  short-leg plus universe SR +1.70. Not a one-sided artifact.
- Yearly net-taker SR 2020–2026: 2.78, 2.48, 0.62, 2.08, 0.31, 2.76, 3.02.
  Positive every year; no single-period concentration; no recent decay.
- Economics: 7d taker-buy share = persistent aggressive-flow imbalance;
  consistent with the batch's structural finding that crypto pays
  attention/continuation premia. Corr 0.38 to price momentum — related
  family, distinct information.

## 4-sleeve combo (construction diagnostic, post hoc)

mom30s1 + K2 + K11 + K12, equal risk, pairwise |corr| ≤ 0.38:
net-taker SR **2.07** full sample, **1.75** 2023+. At a static-leverage
15% vol target: backtest ann ≈ 31%, worst drawdowns proportionally
smaller than the 3-sleeve version.

## Reading

1. **Carry works and is orthogonal to momentum.** K2 (7d funding mean):
   SR 1.05 at taker fees, corr 0.04 with mom30s1. The carry family shows an
   interior turnover optimum: 1d dies on costs, 30d dilutes signal, 7d is
   the sweet spot — same cost-vs-decay logic as the momentum grid.
2. **The entire "short the risky/hyped side" family is wrong-signed
   here.** Low-vol, BAB, short-MAX, short-skew, short-abnormal-volume all
   lost: in this sample the high-vol/high-beta/recently-pumping side
   *outperforms*. Daily reversal is also wrong-signed (continuation
   dominates at 1–3d). This is a coherent structure — crypto cross-section
   pays risk-on/attention premia where equity lore expects the opposite.
   Per protocol these stay rejected; a sign-flipped "high-vol premium"
   candidate would be a NEW look requiring registration, and its economics
   (long junk-beta) need scrutiny before spending it.
3. **Within-venue diversification already lifts Sharpe materially**
   (post-hoc diagnostic, flagged as such): equal-risk mom30s1 + K2 + K11,
   pairwise |corr| ≤ 0.11, net-taker SR **1.59** full sample, **1.20**
   2023+. That is the current best honest estimate of what this venue
   supports with one price factor + one carry factor + one liquidity
   factor at daily rebalance and taker fees.

## Next

- ~~Judge K12 when re-download lands (same bars).~~ *(done same day — verdict
  in the table above; stale bullet struck 2026-07-22.)*
- Beta-neutralize and vol-target the 3-sleeve combo; check residual BTC
  beta — this is the within-sleeve "hedge" that matters first.
- Cross-venue combination (equity/options sleeves) blocked on the other
  ledgers having an accepted factor; crypto L/S corr to those will be
  ~0 by construction and the benefit arrives automatically when they do.
