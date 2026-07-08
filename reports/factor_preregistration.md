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

- Factor 1 standalone monthly IC, **clean corpus** (post contamination-fix
  2026-07-08, commit `0d930e0`): +0.012, t = 2.3. (Was +0.0037/t=0.68 on
  pre-fix contaminated inputs.) Incremental over the full price/volume
  baseline t=1.5 — borderline, below the promotion bar.
- ⚠ The two diagnostics below (sub-period ICs, dollar-volume-half attention
  gradient) were computed on **pre-fix** inputs and are not yet re-run:
  sub-period ICs showed no decay pattern (2012-15: t=−1.5; 2023-26: t=+2.2,
  post-hoc max over 4 buckets); no attention gradient (t=0.5 vs 0.9).
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

> ⚠ **2026-07-08 contamination fix (commit `0d930e0`) supersedes the text-factor
> rows below dated 2026-07-08 R1 and 2026-07-07 factor 15 / factor-1 diagnostics.**
> Those were computed on inputs corrupted by a fallback-extraction switch and
> unfiltered amendments. Rows are kept (they count toward N), but their numbers
> are void. Clean re-runs: factor 1 standalone t=2.3 / incr t=1.5; 10 t=0.68;
> 12 t=0.72; 15 t=0.64 (all sector-neutral ≤0.55). Factors 11/13/14 recomputing.

| When | What | Result |
|------|------|--------|
| 2026-07-08 | **[SUPERSEDED — pre-fix]** R1 full A/B 2012–2026, raw (standalone t / over-baseline t / over-factor-1 t): 1: +0.80/+0.84/— · 10: +1.46/+0.94/+1.12 · 11: +0.23/+0.70/−0.23 · 12: +0.71/+0.51/+0.98 · 13: +1.23/+1.40/+1.34 · 14: −1.02/−0.78/−1.02 | dirty inputs; void — see banner |
| 2026-07-08 | **[SUPERSEDED — pre-fix]** R1 sub-window standalone t (2012-20 / 2021-26): 1: −0.59/+1.89 · 10: +1.80/+0.18 · 11: −0.43/+0.89 · 12: +1.54/−1.14 · 13: +0.35/+1.56 · 14: +0.25/−2.09 | dirty inputs; void — see banner |
| 2026-07-08 | **[SUPERSEDED — pre-fix]** magnitude 3-way: 1 TF-IDF vs 11 finance-embed vs 13 bge-embed | dirty inputs; A/B to be re-run after 11/13 recompute |
| 2026-07-07 | **[SUPERSEDED — pre-fix]** factor 15 combined freshest-filing stream | dirty inputs; clean re-run: standalone t=1.41, incr t=0.64 → still rejected |
| 2026-07-08 | **R1 re-run on clean inputs** (post `0d930e0`): incr-IC t over full baseline — 1: +1.5 · 10: +0.68 · 12: +0.72 · 15: +0.64 (sector-neutral: 1.5/0.25/0.55/0.01) | factor 1 now borderline (was the null); 10/12/15 still spanned by controls; 11/13/14 pending recompute |
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
| 2026-07-08 | **[SUPERSEDED — pre-fix]** R1 sector-neutral diagnostic, full panel window (vs full baseline): 10: +0.45 · 11: +1.26 · 12: +0.33 · 13: +1.54 · 14: −1.22 · 1: +0.64 | dirty inputs; void. Clean sector-neutral: 1: +1.5 · 10: +0.25 · 12: +0.55 · 15: +0.01; 11/13/14 pending |

| 2026-07-08 | event-window IC around filings (quarterly-clustered, EW-adjusted): 10-Q d1-5 t=−0.26 · d6-21 +0.98 · **d22-63 +2.65 (IC +0.030)** ; 10-K control: +0.06 / −1.22 / −1.05 | one cell at the ~25-look noise ceiling; 10-K does NOT replicate the pattern |
| 2026-07-08 | 10-Q d22-63 halves | 2012-19 t=+1.45, 2020-26 t=+2.29 — both positive (only text cell that doesn't sign-flip); confound: window ends at next quarter's earnings |

Hyperparameter variants (e.g. factor-14 MATCH_THRESH, chunk size) get ledger
lines too when swept.
