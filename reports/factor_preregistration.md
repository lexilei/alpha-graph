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

- C1 standalone monthly IC, **clean corpus** (post contamination-fix
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
| Magnitude | how much did it change? | C1 TF-IDF, C3 finance-embed, C5 bge-embed, C7 combined-stream (rejected) |
| Direction | did tone worsen? | C4 LM tone shift |
| Added content | what's genuinely new? | C6 bge change-detect (+ BM25 lexical, if run) |
| (10-Q magnitude) | 10-Q YoY change | C2 cos_10q_yoy |

## Evaluation defaults (reference points, not gates)

- Universe: 499-ticker panel, monthly cross-sections, 2012–2026 (2011 excluded:
  no cosine pairs). Target `fwd_return_21d`; monthly cross-sectional Spearman
  IC via `factor_orthogonality.py`.
- Useful reference splits: 2012–2020 / 2021–2026; sub-period buckets as in the
  diagnostics above. None of these are binding.
- Reference bars when deciding what is *interesting*: standalone t, incremental
  t over the price/volume baseline, and DSR against the ledger N. Thresholds
  are judgment calls made per decision, not fixed in advance.
- The price/volume baseline is {B1 mom21, B2 mom5, B3 vol21, B4 volz, B5 mom12-1,
  B6 log$vol} (B5/B6 added 2026-07-07). `--sector-neutral` demeans all rank-z
  series within sector — report alongside raw as a robustness diagnostic.
  Controls are conditioning variables, not ledger trials.

## Ledger (one line per thing tried — keep appending)

> ⚠ **2026-07-08 contamination fix (commit `0d930e0`) supersedes the text-factor
> rows below dated 2026-07-08 R1 and 2026-07-07 C7 / C1 diagnostics.**
> Those were computed on inputs corrupted by a fallback-extraction switch and
> unfiltered amendments. Rows are kept (they count toward N), but their numbers
> are void. Clean re-runs: C1 standalone t=2.3 / incr t=1.5; C2 t=0.68;
> C4 t=0.72; C7 t=0.64 (all sector-neutral ≤0.55). C3/C5/C6 recomputing.

| When | What | Result |
|------|------|--------|
| 2026-07-08 | **[SUPERSEDED — pre-fix]** R1 full A/B 2012–2026, raw (standalone t / over-baseline t / over-C1 t): C1: +0.80/+0.84/— · C2: +1.46/+0.94/+1.12 · C3: +0.23/+0.70/−0.23 · C4: +0.71/+0.51/+0.98 · C5: +1.23/+1.40/+1.34 · C6: −1.02/−0.78/−1.02 | dirty inputs; void — see banner |
| 2026-07-08 | **[SUPERSEDED — pre-fix]** R1 sub-window standalone t (2012-20 / 2021-26): C1: −0.59/+1.89 · C2: +1.80/+0.18 · C3: −0.43/+0.89 · C4: +1.54/−1.14 · C5: +0.35/+1.56 · C6: +0.25/−2.09 | dirty inputs; void — see banner |
| 2026-07-08 | **[SUPERSEDED — pre-fix]** magnitude 3-way: C1 TF-IDF vs C3 finance-embed vs C5 bge-embed | dirty inputs; A/B to be re-run after C3/C5 recompute |
| 2026-07-07 | **[SUPERSEDED — pre-fix]** C7 combined freshest-filing stream | dirty inputs; clean re-run: standalone t=1.41, incr t=0.64 → still rejected |
| 2026-07-08 | **R1 re-run on clean inputs** (post `0d930e0`): incr-IC t over full baseline — C1: +1.5 · C2: +0.68 · C4: +0.72 · C7: +0.64 (sector-neutral: 1.5/0.25/0.55/0.01) | C1 now borderline (was the null); C2/C4/C7 still spanned by controls; C3/C5/C6 pending recompute |
| 2026-07-09 | **R1 COMPLETE — C3/C5/C6 recomputed clean.** incr over baseline / sector-neutral / over-C1: C3 fin +1.24/+1.96/+0.10 · C5 bge **+1.82/+2.08**/+1.39 · C6 change −1.33/−1.72/−1.23 (standalone −2.06, paper sign) | **A/B flips vs the pre-fix null**: general embedding (C5) > TF-IDF (C1) > finance-tuned (C3); C5 adds real info over C1 (t=1.39), C3 does not (0.10). C5 co-leads with C10. |
| 2026-07-09 | **Multiple-testing tally**: ~8 text/graph factors × 3 modes + A/Bs ≈ 25+ correlated looks. Max incr-t observed 1.97 (f20), 1.82 (f13); max sector-neutral 2.08 (f13) | E[max\|null] for this many correlated looks ≈ 2.0–2.3 → **no factor clears the bar**; the cluster at t≈1.5–2 is noise-compatible. Qualitative A/B structure (13>1>11) is the robust finding. |
| 2026-07-07 | diagnostic: C1 IC by sub-period (4 buckets) | no decay pattern; max bucket t=2.17 (post-hoc) |
| 2026-07-07 | diagnostic: C1 IC by dollar-volume half | no attention gradient (t=0.5 / 0.9) |
| 2026-07-07 | controls: +B5 mom_252_21, +B6 log_dollar_volume, sector-neutral mode | conditioning, not trials |
| 2026-07-08 | C8 spillover_event (full 2011-26 graph, NaN semantics) | standalone t=−0.95, vs baseline t=−0.68, sector-neutral t=+0.12 → nothing, wrong sign |
| 2026-07-08 | C9 spillover_momentum | standalone t=+0.48, incr over baseline t=+1.34 (sign matches C-F), sector-neutral halves to t=+0.64 → noise-compatible |
| 2026-07-08 | corr(18, 19) monthly rank | ~0.00 — the two propagations are independent; no combination rescue |
| 2026-07-08 | C10 spillover_cust_mom registered (customer-only, conf≥0.8, neighbors' 21d momentum) | single variant, no parameter sweep |
| 2026-07-08 | C10 full sample (165m, xs 182) | standalone t=+1.32; incr over baseline **t=+1.97** (IC +0.0142, project's largest); sector-neutral t=+1.54 — survives, unlike C9 |
| 2026-07-08 | C10 split halves (no tuning) | 2012-20: incr t=+1.61 / sn +0.89; 2021-26: +1.13 / sn +1.44 — all 4 cells positive, no post-2020 death |
| 2026-07-08 | C10 vs C9 | incr t=+1.39 — asymmetry carries info beyond the symmetric average |
| 2026-07-08 | C10 quintile L/S | +0.32%/mo, t=+1.69, 55% hit — ~1/5 of C-F's 1981-2004 magnitude, plausible for mega-caps 20y later |
| 2026-07-08 | **[SUPERSEDED — pre-fix]** R1 sector-neutral diagnostic, full panel window (vs full baseline): C2: +0.45 · C3: +1.26 · C4: +0.33 · C5: +1.54 · C6: −1.22 · C1: +0.64 | dirty inputs; void. Clean sector-neutral: C1: +1.5 · C2: +0.25 · C4: +0.55 · C7: +0.01; C3/C5/C6 pending |

| 2026-07-08 | event-window IC around filings (quarterly-clustered, EW-adjusted): 10-Q d1-5 t=−0.26 · d6-21 +0.98 · **d22-63 +2.65 (IC +0.030)** ; 10-K control: +0.06 / −1.22 / −1.05 | one cell at the ~25-look noise ceiling; 10-K does NOT replicate the pattern |
| 2026-07-08 | 10-Q d22-63 halves | 2012-19 t=+1.45, 2020-26 t=+2.29 — both positive (only text cell that doesn't sign-flip); confound: window ends at next quarter's earnings |

| 2026-07-09 | **8-K event round** (new source; 179/499 tickers with 8-K corpus). Tested ~8 constructions vs full baseline (sector-neutral t): frequency **abnormal-z −2.92 (C11)**, unscheduled-abnormal-z −3.04, raw 3mo count −1.21, raw 12mo count +1.59, hand-scored EW severity −1.49, hard-negative-item count +0.07 (C12), unscheduled density −0.99 (C13), LM sentiment +0.93 (C14) | **The signal is filing-frequency abnormality, not content.** C11 verified: split-halves −1.94/−2.89, quintile L/S +0.41%/mo t=2.55, no lookahead → candidate. Item/tone axes dead → C12–C14 rejected. Count level (vs z-score) and hand-scoring don't work. |
| 2026-07-09 | multiple-testing note for the 8-K round | ~8 correlated activity/content/tone constructions; the winner (C11 −2.92) and its unscheduled variant (−3.04) are the same idea. Treat as one axis with a winner, per protocol. Full-499 re-verification pending backfill. |

| 2026-07-10 | **Graph propagation + topology round** (7 constructions on the customer/supplier/competitor graph; vs full baseline, sector-neutral t): #1 customer-abnormal-8K-freq +0.74 · #2 customer-text(bge) +1.01 · #3 customer-volatility +0.60 · #4 supplier-momentum-upstream +1.85 · #5 competitor-momentum +0.03 · #7 customer-concentration-HHI −2.87 · #8 degree-centrality +1.62 | **Nothing survives.** #7 (HHI) was the slow-moving-factor autocorrelation trap: naive −2.87 → HAC(L=12/24) t≈−1.6, split-halves −2.39/−1.25, and 0.67-correlated with customer count (ncust itself sn +1.81) — a connectivity proxy. #3 dies under sector-neutralization (standalone 2.68 → sn 0.60 = sector effect). #4 (upstream momentum) borderline but below ceiling. **#1 pending full 8-K coverage** — it composes the two strongest ideas (C11 abnormal frequency × C10 customer channel), currently only 22% coverage. |

| 2026-07-10 | **LightGBM combiner, tuned properly** (pre-declared 9-config grid — leaves {7,15,31} × train-window {12,36,60}mo — selected on IS 2012-2020 common eval window 2017-2020, single-shot OOS 2021-2026). IS winner leaves=31/window=60m at IS Sharpe 1.87 (E[max\|null,9]=0.76) | **OOS gross 0.98 / net ~0.62 (30bp/mo), monthly t=2.24 (63mo). IS→OOS shrinkage ~50%.** Attribution (same config, OOS): baseline-only **1.02** ≥ full 0.98; novel-SEC-factors-only 0.28. The combiner is a momentum machine; the SEC factors contribute ≈ zero at portfolio level OOS. Window length (12→60mo) was the material knob, not tree complexity. |

| 2026-07-10 | **Full-corpus retest** (8-K backfill complete: 499 tickers, 100,618 filings). Sector-neutral incr t: C12 hard-items +0.08 · C13 unsched +0.24 · C14 sentiment +0.08 · unsched-abnormal-z −1.15 (was −3.04 on the 179-name subset) · composite customer-propagated-freq +0.90 (halves +0.79/+0.55, sign opposite to prediction) | C12–C14 rejections confirmed at full coverage. The unscheduled-frequency variant attenuates like C11 itself. The composite (C11 signal over C10's customer channel) does not work — the propagation round closes with nothing surviving. |

| 2026-07-10 | **Adversarial code review of the 8-K pipeline** (agent, read-only, 27 tool calls). Verdicts: C11 STANDS (no lookahead — excluding all last-calendar-day filings strengthens it to −2.94; IC broad-based, NW t=−3.22, drop-5-best −1.99), C12–C14 nulls STAND (the retest's same-month merge leak is ≤0.4% and inflationary-only), composite rejection STANDS. New characterization: C11 is a fresh-month phenomenon — fresh-attach months t=−3.68, stale months +0.35, one-month lag +1.04; decays within one month. Doc bugs fixed: split-half boundary dropped 2019 (−2.17 was 2012-18; with 2019 it is −2.07); C13's −3.04 parenthetical was subset-era (full corpus −1.15). | Review also ran one extra variant (counts toward N): hard-item abnormal-z, sn t=−0.19 standalone / +0.89 over baseline+C11 — null, reinforcing the content-axis rejection. Actionable: a daily within-month count would carry the −3.68-grade signal. |

| 2026-07-11 | **Item 1 (PIT membership) implemented** (plan v2). Curated rename map: 31 old symbols -> 18 panel tickers incl. flattened chains (SYMC/NLOK->GEN, HCP/PEAK->DOC, CBS/VIAC/PARA->PSKY, MWV/WRK->SW) and one dated reuse rule (IR<=2020-02 -> TT). Dataset facts: older renames arrive back-normalized (WLP->ANTM, AA->ARNC, GOOGL from 2010); SQ never listed (Block joined 2025-07 as XYZ, replacing HES); FISV/FI double-listed (deduped). 5 panel tickers never match (post-CSV 2026 joiners: CIEN COHR LITE SATS VRT) — excluded from PIT mode. Mask removes 331,087 rows (18.5%). Gates pass: META first PIT row 2013-12-23, TSLA 2020-12-21. | Survivor coverage by year (forward-bias fix ONLY, departed names unrecoverable): 2011 61.0% (gap 194) -> 2015 67.3% -> 2020 82.7% -> 2025 95.4% (gap 23). PIT deltas will be attributed in the 2^3 factorial with items 2+4; pit_universe defaults False until the v0 freeze. |

Hyperparameter variants (e.g. C6 MATCH_THRESH, chunk size) get ledger
lines too when swept.
