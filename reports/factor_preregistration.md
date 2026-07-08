# Factor Pre-Registration — 10-K/10-Q text family

Written **before** the full-corpus A/B and incremental-IC results are examined
(embedding factors still computing at time of writing). Its purpose is to fix
the decision rules and the trial count in advance so the eventual selection is
not a multiple-testing artifact — the failure mode that produced this project's
retracted Sharpe 1.77 / 1.91 / Method-E numbers.

Refer to factors by their permanent IDs (see `FACTORS.md`).

## Economic hypothesis (one, not many)

Lazy Prices (Cohen-Malloy-Nguyen 2020): deliberate change in a firm's periodic
filings predicts subsequent returns. All text factors below are **different
measurements of this single hypothesis**, not independent bets.

## Prior: LOW

Audited evidence in this universe already says the hypothesis barely holds:
signal ≈ 12-1 momentum, the decile relation is U-shaped (short side doesn't
replicate), and it decayed to ~0 post-2020. On the full corpus, factor 1's
standalone monthly IC is t = +0.68 (insignificant). Because the prior is low,
the bar to *believe* a factor is raised (see thresholds), not the default t=2.

## The family, by axis

One hypothesis, three measurement axes. Within-axis variants are collapsed to
ONE by A/B; every variant tried still counts toward N.

| Axis | Question | Factors (variants) |
|------|----------|--------------------|
| Magnitude | how much did it change? | 1 TF-IDF, 11 finance-embed, 13 bge-embed |
| Direction | did tone worsen? | 12 LM tone shift |
| Added content | what's genuinely new? | 14 bge change-detect (+ 15 BM25 lexical, if run) |
| (10-Q magnitude) | 10-Q YoY change | 10 cos_10q_yoy |

## Data / evaluation protocol (fixed)

- Universe: the 499-ticker panel (`market_data`), monthly cross-sections.
- Period: **2012–2026** (2011 excluded — first year, no cosine pairs).
- Target: `fwd_return_21d`. Metric: cross-sectional Spearman IC, aggregated to a
  monthly IC series; t = mean/(sd/√n) over months (`factor_orthogonality.py`).
- **In-sample vs OOS split (pre-registered): IS = 2012–2020, OOS = 2021–2026.**
  Selection happens on IS only; OOS is touched once, for confirmation.
- Regime factors 4/5 are `inactive` (market-level, no cross-sectional content) —
  not in the pool.

## Decision rule (candidate → accepted)

A factor is promoted only if ALL hold:

1. **In-sample incremental-IC t > 3** over the current accepted set (raised from
   2 for the low prior + multiple testing).
2. **OOS incremental-IC t > 2** (confirmation on 2021–2026 — the deployment-
   relevant, post-decay period).
3. **Deflated Sharpe / IC** using the honest trial count N below > 0.90.

Within an axis: run the A/B on IS, keep the single best-incremental-IC variant,
discard the rest (they remain `candidate`/`rejected` in FACTORS.md, never
silently dropped). Selection is by incremental IC + economic structure, never by
chasing the max raw number.

## Trial ledger (N = everything tried) — fill as we go

| Round | Factors evaluated | Axis | Variants tried | Kept |
|-------|-------------------|------|----------------|------|
| R1 | 1, 11, 13 | magnitude | 3 | TBD |
| R1 | 12 | direction | 1 | TBD |
| R1 | 14 (+15?) | added content | 1–2 | TBD |
| R1 | 10 | 10-Q magnitude | 1 | TBD |
| R1 | 15 | magnitude (combined freshest-filing stream) | 1 | no — IS incr-IC t=0.64 (<3), standalone t=0.83 |

Factor 15 added 2026-07-07 **before its evaluation**: the paper-faithful
combined signal — each stock's most recent periodic filing's YoY same-type
cosine (union of factors 1+10, raw scores pooled). Same axis, same decision
rule as the family; its variants count toward N like any other.

Hyperparameter variants also count (e.g. factor-14 MATCH_THRESH, chunk size).
**N_total for this round is the sum of the "variants tried" column** and feeds
the DSR in rule 3. Record the final N here once the round closes.

## What we will NOT do

- Add a 4th magnitude encoder hoping for a higher number (N inflation).
- Report a within-axis winner's raw IC without the DSR/N discount.
- Choose the IS/OOS split or the t-threshold after seeing results.
- Re-run on a sub-window where a factor happens to work (the 1.77/1.91 mistake).

## Sign-off

Committed before results: ____ (git commit hash of this file).
Round closed / N_total: ____.
