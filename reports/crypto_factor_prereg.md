# Crypto perp factor pre-registration — batch K1 (2026-07-21)

Committed before any candidate below was evaluated. Protocol is identical to
the feasibility scan (`crypto_perp_xs_feasibility.md`): top-100 liquidity
universe, quintile L/S equal weight, 1.0 gross/side, signal at close t earns
close t → close t+1, funding P&L included, fee ladder {0, 2bp, 5bp}.
Panel: 787 USDT perps 2020-01 .. 2026-06, survivorship-free.

Baseline for incrementality: `mom30s1` = momentum lb30 skip1 (best prior
cell; itself only one look among six — treated as reference, not truth).

## Candidates (definitions and signs frozen)

Signal s is "long-attractiveness"; wrong realized sign = reject, no re-signing.

| ID | name | s at close t | rationale |
|----|------|--------------|-----------|
| K1 | carry_1d | − funding summed over last 1d | crowded longs pay; short them |
| K2 | carry_7d | − mean daily funding, 7d | slower carry |
| K3 | carry_30d | − mean daily funding, 30d | slowest carry |
| K4 | rev_1d | − ret(1d) | short-term reversal |
| K5 | rev_3d | − ret(3d) | short-term reversal |
| K6 | lowvol_30d | − std(daily ret, 30d) | low-vol anomaly |
| K7 | bab_60d | − beta vs BTCUSDT, 60d | betting against beta |
| K8 | max_30d | − max(daily ret, 30d) | lottery demand, short spikes |
| K9 | skew_30d | − skew(daily ret, 30d) | skew preference |
| K10 | size_qv | − log(median qvol, 30d) | small-size premium |
| K11 | amihud_30d | + mean(abs ret / qvol, 30d) | illiquidity premium |
| K12 | tbr_7d | + mean(taker_buy_vol / vol, 7d) | flow imbalance, follow buyers |
| K13 | volz_7d | − (mean qvol 7d / mean qvol 90d) | attention/overpricing, short hype |
| K14 | high_90d | + close / max(close, 90d) | nearness-to-high |

## Decision statistics (pinned)

Primary: net-taker L/S Sharpe, full sample. Secondary: net-taker Sharpe
2023+, daily gross-return correlation with mom30s1, ΔSR of an equal-risk
50/50 combo with mom30s1 (full-sample vols; diagnostic, mild lookahead,
noted) versus mom30s1 alone (both net-taker).

## Bars (pinned)

- **Register-worthy standalone**: net-taker SR ≥ 0.5, t = SR×√yrs ≥ 2.0,
  2023+ same sign. t = 2.0 is below the 14-look family expected-max ceiling
  (≈ 2.2σ under the null), so this tier is "candidate", not "accepted".
- **Register-worthy diversifier**: |corr with mom30s1| ≤ 0.4 AND combo
  ΔSR ≥ +0.10 AND standalone net-taker SR ≥ 0.3.
- Wrong pre-committed sign → rejected regardless of magnitude.
- Look ledger for this venue: 6 (feasibility grid) + 14 (this batch) = 20.

Anything promoted later to live capital must be re-registered with its own
pinned bar on data that postdates this commit.
