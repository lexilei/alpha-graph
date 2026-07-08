# Factor Research Log — 10-K/10-Q text family

Two modes, by decision 2026-07-07 (supersedes the locked pre-registration in
this file's history, commit 5d4210a):

- **Explore (default, unrestricted).** Any window, any variant, any threshold,
  any diagnostic. Numbers produced here are hypotheses, not results — they may
  be quoted internally but not claimed externally.
- **Claim (only when quoting a number outside — README, interview, report).**
  At that moment, design a one-shot honest confirmation: data not used to find
  the effect, and N = the ledger count below. Until then nothing is "confirmed".

The only standing habit is the **ledger**: one line per thing tried. It
restricts nothing; it preserves the option to make a strong claim later
(an unrecorded N is unrecoverable).

Refer to factors by their permanent IDs (see `FACTORS.md`).

## Economic hypothesis (one, not many)

Lazy Prices (Cohen-Malloy-Nguyen 2020): deliberate change in a firm's periodic
filings predicts subsequent returns. All text factors are different
measurements of this single hypothesis, not independent bets.

## Context (verified on our artifacts)

- Factor 1 standalone monthly IC on the complete corpus: +0.0037, t = 0.68.
- Sub-period ICs show no decay pattern (2012-15: t=−1.5; 2023-26: t=+2.2 — the
  latter is a post-hoc max over 4 buckets, not significant).
- No attention gradient within the S&P 500 (low- vs high-dollar-volume halves:
  t=0.5 vs 0.9).
- Universe caveat: 306 of 798 tickers that passed through the index 2012–2026
  (38%) are absent (departed members) — every IC here is measured on a
  survivor-tilted cross-section, which attenuates any true signal.

## The family, by axis

| Axis | Question | Factors (variants) |
|------|----------|--------------------|
| Magnitude | how much did it change? | 1 TF-IDF, 11 finance-embed, 13 bge-embed, 15 combined-stream (rejected) |
| Direction | did tone worsen? | 12 LM tone shift |
| Added content | what's genuinely new? | 14 bge change-detect (+ BM25 lexical, if run) |
| (10-Q magnitude) | 10-Q YoY change | 10 cos_10q_yoy |

## Evaluation defaults (reference points, not gates)

- Universe: 499-ticker panel, monthly cross-sections, 2012–2026 (2011 excluded:
  no cosine pairs). Target `fwd_return_21d`; monthly cross-sectional Spearman
  IC via `factor_orthogonality.py`.
- Useful reference splits: 2012–2020 / 2021–2026; sub-period buckets as in the
  diagnostics above. None of these are binding.
- Reference bars when deciding what is *interesting*: standalone t, incremental
  t over the price/volume baseline, and DSR against the ledger N. Thresholds
  are judgment calls made per decision, not fixed in advance.
- The price/volume baseline is {6 mom21, 7 mom5, 8 vol21, 9 volz, 16 mom12-1,
  17 log$vol} (16/17 added 2026-07-07). `--sector-neutral` demeans all rank-z
  series within sector — report alongside raw as a robustness diagnostic.
  Controls are conditioning variables, not ledger trials.

## Ledger (one line per thing tried — keep appending)

| When | What | Result |
|------|------|--------|
| R1 | factors 1, 11, 13 (magnitude A/B) | pending factor-14 compute completion |
| R1 | factor 12 (direction) | pending |
| R1 | factor 14 (added content) | pending |
| R1 | factor 10 (10-Q magnitude) | pending |
| 2026-07-07 | factor 15 combined freshest-filing stream | IS incr-IC t=0.64, standalone t=0.83 → rejected |
| 2026-07-07 | diagnostic: factor-1 IC by sub-period (4 buckets) | no decay pattern; max bucket t=2.17 (post-hoc) |
| 2026-07-07 | diagnostic: factor-1 IC by dollar-volume half | no attention gradient (t=0.5 / 0.9) |
| 2026-07-07 | controls: +16 mom_252_21, +17 log_dollar_volume, sector-neutral mode | conditioning, not trials |

Hyperparameter variants (e.g. factor-14 MATCH_THRESH, chunk size) get ledger
lines too when swept.
