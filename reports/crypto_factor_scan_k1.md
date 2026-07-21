# Crypto factor batch K1 — results

2026-07-21. Evaluation of the batch frozen in `crypto_factor_prereg.md`
(committed before this run). 13 of 14 evaluated; **K12 pending** — the
first klines download dropped `taker_buy_volume`, re-download running;
K12 will be judged under the same pinned bars when it lands. Numbers:
`crypto_factor_scan_k1.csv`. Baseline mom30s1 net-taker SR 0.946 / 6.3y.

## Verdicts

| ID | net-taker SR | t | 2023+ | corr mom | combo ΔSR | verdict |
|----|---:|---:|---:|---:|---:|---|
| K2 carry_7d | **1.05** | **2.63** | 0.58 | 0.04 | **+0.45** | PASS standalone+diversifier |
| K14 high_90d | 0.93 | 2.30 | 0.88 | 0.56 | +0.18 | PASS standalone (momentum cousin) |
| K11 amihud_30d | 0.68 | 1.71 | 0.34 | −0.00 | +0.21 | PASS diversifier |
| K3 carry_30d | 0.65 | 1.64 | 0.62 | 0.09 | +0.15 | PASS diversifier (corr 0.55 w/ K2) |
| K1 carry_1d | 0.19 | — | −0.06 | | | reject: turnover 0.87/d eats it |
| K10 size_qv | 0.47 | 1.18 | −0.42 | | | reject: dead 2023+ |
| K4 rev_1d, K5 rev_3d | −0.90, −1.06 | | | | | REJECT wrong sign |
| K6 lowvol, K7 bab, K8 max, K9 skew, K13 volz | −0.32 … −1.22 | | | | | REJECT wrong sign |

Family context: 13 looks this batch, venue ledger now 19 looks (20 with
K12). Expected-max |t| under the null for 19 looks ≈ 2.3σ. K2 (2.63) is the
only candidate above the family ceiling; K14 (2.30) sits at it.

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

- Judge K12 when re-download lands (same bars).
- Beta-neutralize and vol-target the 3-sleeve combo; check residual BTC
  beta — this is the within-sleeve "hedge" that matters first.
- Cross-venue combination (equity/options sleeves) blocked on the other
  ledgers having an accepted factor; crypto L/S corr to those will be
  ~0 by construction and the benefit arrives automatically when they do.
