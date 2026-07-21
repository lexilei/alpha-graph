# Crypto perp factor pre-registration — batch M (2026-07-21)

Committed before the metrics dataset existed locally (download launched the
same evening; this file is frozen first). Protocol, universe, portfolio
construction, fee ladder and decision statistics identical to batch K
(`crypto_factor_prereg.md`). Baseline for incrementality: the 3-sleeve combo
(mom30s1 + K2 + K11) rather than mom30s1 alone — a new factor must now add
to what we already hold. Data: `perp_metrics_daily.parquet`, last-snapshot
(_last) and day-mean (_mean) aggregates of Binance UM metrics; sample starts
where metrics coverage starts (~2022), shorter than batch K.

Sign priors use batch-K's structural lesson (crypto cross-section pays
risk-on/continuation, punishes crowded-retail positioning). That lesson came
from klines/funding data; this batch is judged on a distinct dataset.

## Candidates (definitions and signs frozen)

OI_t = sum_open_interest_value_last at day t.

| ID | name | s at close t | rationale |
|----|------|--------------|-----------|
| M1 | oi_chg_7d | + log(OI_t) − log(OI_{t−7}) | new-money continuation |
| M2 | oi_crowd | − OI_t / median(qvol, 30d) | crowded leverage → squeeze tail |
| M3 | lsr_retail_7d | − mean(count_long_short_ratio_last, 7d) | fade retail net-long |
| M4 | lsr_top_7d | + mean(sum_toptrader_long_short_ratio_last, 7d) | follow top-trader positioning |
| M5 | oi_chg_1d | + log(OI_t) − log(OI_{t−1}) | fast new-money continuation |

Noted overlap: taker long/short flow from metrics duplicates K12 (klines
taker share); not registered here to avoid a double look on one idea.

## Decision statistics and bars (pinned, as batch K except baseline)

Primary net-taker L/S SR; secondary 2023+, corr with the 3-sleeve combo,
equal-risk combo ΔSR vs the 3-sleeve combo. Standalone: SR ≥ 0.5, t ≥ 2.0,
2023+ same sign. Diversifier: |corr| ≤ 0.4, ΔSR ≥ +0.10, SR ≥ 0.3.
Wrong sign → reject, no re-signing. Venue look ledger: 20 (K incl. pending
K12) + 5 = 25; expected-max |t| under the null ≈ 2.4σ at 25 looks.
