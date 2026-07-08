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
| 2026-07-08 | R1 full A/B 2012–2026, raw (standalone t / over-baseline t / over-factor-1 t): 1: +0.80/+0.84/— · 10: +1.46/+0.94/+1.12 · 11: +0.23/+0.70/−0.23 · 12: +0.71/+0.51/+0.98 · 13: +1.23/+1.40/+1.34 · 14: −1.02/−0.78/−1.02 | max \|t\|=1.46 across ~18 looks — nothing clears 2 |
| 2026-07-08 | R1 sub-window standalone t (2012-20 / 2021-26): 1: −0.59/+1.89 · 10: +1.80/+0.18 · 11: −0.43/+0.89 · 12: +1.54/−1.14 · 13: +0.35/+1.56 · 14: +0.25/−2.09 | signs flip across halves for 4/6 — noise signature |
| 2026-07-08 | magnitude 3-way: 1 TF-IDF vs 11 finance-embed vs 13 bge-embed | no encoder beats bag-of-words (11 worst +0.23; incrementals over 1 all \|t\|<1.4) |
| 2026-07-07 | factor 15 combined freshest-filing stream | IS incr-IC t=0.64, standalone t=0.83 → rejected |
| 2026-07-07 | diagnostic: factor-1 IC by sub-period (4 buckets) | no decay pattern; max bucket t=2.17 (post-hoc) |
| 2026-07-07 | diagnostic: factor-1 IC by dollar-volume half | no attention gradient (t=0.5 / 0.9) |
| 2026-07-07 | controls: +16 mom_252_21, +17 log_dollar_volume, sector-neutral mode | conditioning, not trials |
| 2026-07-08 | factor 18 spillover_event (full 2011-26 graph, NaN semantics) | standalone t=−0.95, vs baseline t=−0.68, sector-neutral t=+0.12 → nothing, wrong sign |
| 2026-07-08 | factor 19 spillover_momentum | standalone t=+0.48, incr over baseline t=+1.34 (sign matches C-F), sector-neutral halves to t=+0.64 → noise-compatible |
| 2026-07-08 | corr(18, 19) monthly rank | ~0.00 — the two propagations are independent; no combination rescue |
| 2026-07-08 | factor 20 spillover_cust_mom registered (customer-only, conf≥0.8, neighbors' 21d momentum) | single variant, no parameter sweep |
| 2026-07-08 | factor 20 full sample (165m, xs 182) | standalone t=+1.32; incr over baseline **t=+1.97** (IC +0.0142, project's largest); sector-neutral t=+1.54 — survives, unlike 19 |
| 2026-07-08 | factor 20 split halves (no tuning) | 2012-20: incr t=+1.61 / sn +0.89; 2021-26: +1.13 / sn +1.44 — all 4 cells positive, no post-2020 death |
| 2026-07-08 | factor 20 vs factor 19 | incr t=+1.39 — asymmetry carries info beyond the symmetric average |
| 2026-07-08 | factor 20 quintile L/S | +0.32%/mo, t=+1.69, 55% hit — ~1/5 of C-F's 1981-2004 magnitude, plausible for mega-caps 20y later |

Hyperparameter variants (e.g. factor-14 MATCH_THRESH, chunk size) get ledger
lines too when swept.
