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

## Current accounting (2026-07-13)

- **Counting rule.** N = every ledger row that records a computed evaluation
  statistic — including SUPERSEDED rows (they were looks; voiding a number
  does not un-spend it) — excluding pure registration lines and
  infrastructure/bug-fix notes.
- **Count.** 64 ledger rows as of today (including the 19 factorial cells,
  the 2 v0 re-evals, and the 12 back-filled 2026-07-10 IC-decay-sweep looks
  below). Excluded: 8 — 1 pure registration (C10,
  2026-07-08); 5 infrastructure/bug-fix rows (B5/B6+sector controls
  2026-07-07 "conditioning, not trials"; PIT implementation 2026-07-11;
  ic_tools/pit_universe code review 2026-07-11; RENAME_MAP audit 2026-07-11;
  build_graph dedup fix 2026-07-13); 2 multiple-testing accounting notes
  (2026-07-09, tallies of already-counted looks). **N = 56.**
- **Ceiling.** Selection here is over |t| (signs are not pre-registered), so
  the bar is the two-sided ceiling `alpha_graph.eval.ic_tools.emax_null(2N)`
  = emax_null(112) = **2.57**. The counting judgment calls barely move it:
  counting all 64 rows gives 2.62; also dropping the two borderline counted
  rows (the superseded 3-way A/B, which records no number of its own, and
  the C8–C9 factor-correlation diagnostic, which touches no returns) gives
  2.56.
- **Binding rule.** This ceiling is the significance bar for ANY external
  claim from this ledger. Effective-N (correlation-based, ~7-9 independent
  bets among the factors) describes redundancy only and is never the
  denominator.

As of this date no factor clears the bar (strongest: C15 at v0 −2.29 vs
ceiling ≈2.57).

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

| 2026-07-11 | **Adversarial code review of ic_tools + pit_universe** (agent, 27 tool calls, all findings verified by running). Two HIGH: (F1) PIT mask ran BEFORE feature computation — re-entrant names' lookbacks spanned membership gaps (AMD re-entry momentum read +297% vs true +11%; 40,980 mom_252_21 rows wrong) — mask moved after features, re-entrant rows now bit-identical to non-PIT; (F2) FISV->FI map entry was INVERTED, silently dropping Fiserv for 15y (the META failure class; the gate only tested META) — fixed to FI->FISV, PIT panel 493->494 tickers. Plus: deflated_t now uses the two-sided ceiling emax_null(2N) for \|t\| selection; hac_lags override floored at the overlap minimum; DatetimeIndex split-half boundary coerced to period end; effective_n relabeled (true Li-Ji + capped variant + Nyholt over all M eigenvalues ddof=1, sparse-pair count exposed); quantile_ls tie-straddle warning; default_lags cap-truncation warning; hac_tstat degenerate-series -> NaN; ic_decay implemented. | Hard tests added: map-target-in-panel-vocabulary (would have caught FISV immediately), never-matched == known 2026 joiners, FISV 2015+2024 membership, two-sided deflation calibration (p=0.5 at ceiling), override-guardrail, hand-built exact turnover/break-even, DatetimeIndex boundary, 60/61-day carry edge. 55 tests pass. Clean-verified by the review: HAC==statsmodels to 1e-9, emax_null within 0.05 of simulation, binary search 0/5,438 property-test misses, META/TSLA gates genuinely bite. |

| 2026-07-11 | **RENAME_MAP forensic audit** (agent, 26 tool calls; identity via filing text + price-seam/level forensics). 28/31 entries CONFIRMED (incl. IR boundary exact, MMC->MRSH, PSKY chain, DISH->SATS correctly absent — mapping it would be a 6.2y false grant to old-EchoStar's line). 3 WRONG: FISV inversion (fixed by the parallel code review, verified end-to-end here); **MWV->SW and WRK->SW deleted — panel SW is the Smurfit Kappa price line (never a member; 2011 level ~9.2 vs WestRock ~26+, OTC stale fills), the mapping granted a never-member 14.1y / 3,333 rows of false membership with junk prices**. 2 MISSING added: **PX->LIN** (Praxair listing continued as Linde; 1,910 rows / 8.4y falsely excluded) and **DWDP->DD** (438-row interior hole 2017-08..2019-06). | Back-normalization boundary established: pure renames normalized upstream through ~2018-06; merger-successions NEVER normalized (produced all 4 errors) and require price-seam identity checks. Side flags for later: panel DD +17.8% on 2019-06-03 looks like a missing Corteva ex-distribution price adjustment (data-quality, task-level); filings/EXE holds pre-bankruptcy Chesapeake text under the current ticker (text-factor identity hazard); CSV double-lists KORS+CPRI 2016-2018. |

| 2026-07-13 | **build_graph dedup nondeterminism fixed.** The relationships parquet has 290 same-day duplicate groups (same source/target/filing_date, differing relation/confidence); the dedup's unstable single-key sort made the survivor depend on input row order, so the monthly and daily C10 builders got different edges — 1,217/29,048 month-end value mismatches. Fixed with a stable total order (filing_date, then confidence, then relation — latest/highest/lexicographically-last wins); regression test pins order-independence; both C10 caches rebuilt (grid invariance now 0/29,062). | Prior C10 numbers were computed on an arbitrary tie order — deltas are tie-edge-sized; full recompute lands with the factorial. |

| 2026-07-13 | factorial 1/19: C1 vs BASELINE, sector-neutral, pit=F lag=0 grid=monthly | sn incr t=+1.48 (IC +0.0066), sn raw t=+2.11, 168m, xs 451, cov 89.8% — reproduces clean-C1 ledger 1.5 |
| 2026-07-13 | factorial 2/19: C1 vs BASELINE, sn, pit=F lag=1 grid=monthly | sn incr t=+1.47 (IC +0.0065), sn raw t=+2.08, 168m, xs 451, cov 89.7% — lag ≈ inert for C1 on monthly sampling |
| 2026-07-13 | factorial 3/19: C1 vs BASELINE, sn, pit=T lag=0 grid=monthly | sn incr t=+0.29 (IC +0.0014), sn raw t=+0.74, 168m, xs 382, cov 92.5% |
| 2026-07-13 | factorial 4/19: C1 vs BASELINE, sn, pit=T lag=1 grid=monthly | sn incr t=+0.24 (IC +0.0011), sn raw t=+0.70, 168m, xs 382, cov 92.5% — **v0 cell** (C1 grid-inert: eval inputs verified bit-identical on the daily panel) |
| 2026-07-13 | factorial 5/19: C10 vs BASELINE, sn, pit=F lag=0(≡1, verified) grid=monthly | sn incr t=+1.40 (IC +0.0093), sn raw t=+1.68, 165m, xs 178, cov 34.7% — post-dedup-rebuild (was +1.54 pre-fix; tie-edge-sized delta as predicted) |
| 2026-07-13 | factorial 6/19: C10 vs BASELINE, sn, pit=T lag=0(≡1) grid=monthly | sn incr t=+1.05 (IC +0.0076), sn raw t=+1.01, 165m, xs 160, cov 38.1% |
| 2026-07-13 | factorial 7/19: C10 vs BASELINE, sn, pit=F lag=0 grid=daily | sn incr t=+1.44 (IC +0.0097), sn raw t=+1.57, 166m, xs 177, cov 34.8% |
| 2026-07-13 | factorial 8/19: C10 vs BASELINE, sn, pit=F lag=1 grid=daily | sn incr t=+1.05 (IC +0.0070), sn raw t=+1.23, 165m, xs 178, cov 34.7% |
| 2026-07-13 | factorial 9/19: C10 vs BASELINE, sn, pit=T lag=0 grid=daily | sn incr t=+0.91 (IC +0.0064), sn raw t=+1.02, 166m, xs 160, cov 38.2% |
| 2026-07-13 | factorial 10/19: C10 vs BASELINE, sn, pit=T lag=1 grid=daily | sn incr t=+0.21 (IC +0.0015), sn raw t=+0.33, 165m, xs 161, cov 38.1% — **v0 cell** |
| 2026-07-13 | factorial 11/19: C11 vs BASELINE, sn, pit=F (lag+grid verified inert for C11) | sn incr t=−2.81 (IC −0.0110), sn raw t=−3.20, 168m, xs 458, cov 96.0% — reproduces the registry headline exactly |
| 2026-07-13 | factorial 12/19: C11 vs BASELINE, sn, pit=T (lag+grid inert) | sn incr t=−2.00 (IC −0.0089), sn raw t=−2.42, 168m, xs 386, cov 97.4% — **v0-convention C11** |
| 2026-07-13 | factorial 13/19: C15 vs BASELINE, sn, pit=F lag=0 grid=daily | sn incr t=−3.29 (IC −0.0138), sn raw t=−3.47, 168m, xs 458, cov 93.1% — exceeds prior max headline \|t\| (C11 −2.81); pit=F cell, attribution only |
| 2026-07-13 | factorial 14/19: C15 vs BASELINE, sn, pit=F lag=1 grid=daily | sn incr t=−3.67 (IC −0.0156), sn raw t=−3.72, 168m, xs 458, cov 93.1% — exceeds prior max \|t\|; pit=F cell; t+1 strengthens C15 |
| 2026-07-13 | factorial 15/19: C15 vs BASELINE, sn, pit=T lag=0 grid=daily | sn incr t=−1.85 (IC −0.0090), sn raw t=−2.35, 168m, xs 386, cov 95.0% |
| 2026-07-13 | factorial 16/19: C15 vs BASELINE, sn, pit=T lag=1 grid=daily | sn incr t=−2.29 (IC −0.0111), sn raw t=−2.63, 168m, xs 386, cov 95.0% — **v0 cell**, C15's first headline |
| 2026-07-13 | factorial 17/19 (sensitivity): C1 vs BASELINE, sn, v0 panel + controls lagged 1 td (lag_controls) | sn incr t=+0.10 (IC +0.0005), sn raw t=+0.70, 168m, xs 382 — control timing moves C1 −0.14 |
| 2026-07-13 | factorial 18/19 (sensitivity): C10 vs BASELINE, sn, v0 panel + controls lagged | sn incr t=−0.05 (IC −0.0003), sn raw t=+0.34, 165m, xs 161 — moves C10 −0.26, through zero |
| 2026-07-13 | factorial 19/19 (sensitivity): C15 vs BASELINE, sn, v0 panel + controls lagged | sn incr t=−2.33 (IC −0.0112), sn raw t=−2.63, 168m, xs 386 — C15 insensitive to control timing (−0.04) |
| 2026-07-13 | v0 re-eval: C5 embed_sim_10k_bge vs BASELINE, sn, saved v0 panel (pit=T lag=1 grid=daily) | sn incr t=+1.43 (IC +0.0065, ICIR +0.11), sn raw t=+1.58 (IC +0.0074), 168m, xs 382, cov 92.5% — was sn +2.08 pre-PIT (2026-07-09); same PIT attenuation as C1; below the 1.5 borderline bar → rejected under v0 |
| 2026-07-13 | v0 re-eval: C6 new_content_frac vs BASELINE, sn, saved v0 panel | sn incr t=−0.73 (IC −0.0034, ICIR −0.06), sn raw t=−0.57 (IC −0.0029), 168m, xs 382, cov 92.5% — was sn −1.72 pre-PIT (2026-07-09) → rejected under v0 |

> **Back-fill 2026-07-13 (pre-v0 looks, dated to when they were run).** The
> 2026-07-10 README (`ac5f136`, "Holding-horizon structure") published an
> IC-decay sweep — two families across the 5–126 trading-day horizon grid
> (5/10/21/42/63/126, the grid later formalized as `ic_tools.ic_decay`) —
> but the per-cell looks never got ledger rows. They were looks under the
> counting rule; the 12 rows below repair that. Only two cells' numbers were
> preserved in the README (text @63d, C10 @10d); the rest were published as
> curve shape only (text: monotonic rise to 3–6 months; C10: peak near 10d,
> decay after). All are pre-PIT / pre-v0 attribution-era numbers.

| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 1/12: text family (C5 general-embedding quoted) @ 5d | published as curve shape only (monotonic rise); numeric HAC t not preserved |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 2/12: text family @ 10d | published as curve shape only; numeric HAC t not preserved |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 3/12: text family @ 21d | published as curve shape only; numeric HAC t not preserved |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 4/12: text family @ 42d | published as curve shape only; numeric HAC t not preserved |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 5/12: text family @ 63d | HAC incremental t ≈ +2.7 (the README's published number) |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 6/12: text family @ 126d | published as curve shape only ("rises monotonically out to a 3–6 month horizon"); numeric HAC t not preserved |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 7/12: C10 customer momentum @ 5d | published as curve shape only (peak near 10d); numeric HAC t not preserved |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 8/12: C10 customer momentum @ 10d | HAC incremental t ≈ +3.1 (published; also quoted in the README's C10 table) |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 9/12: C10 customer momentum @ 21d | published as curve shape only (decays after 10d); numeric HAC t not preserved |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 10/12: C10 customer momentum @ 42d | published as curve shape only; numeric HAC t not preserved |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 11/12: C10 customer momentum @ 63d | published as curve shape only; numeric HAC t not preserved |
| 2026-07-10 | (back-filled 2026-07-13; pre-v0) decay sweep 12/12: C10 customer momentum @ 126d | published as curve shape only; numeric HAC t not preserved |
| 2026-07-13 | **Gate-2 tradeable backtest, C15 decile L/S, registered pre-run**: long lowest-z decile, short highest-z decile (sign per C15's negative IC); monthly rebalance at each month's last trading close using that day's available signal; execution at the rebalance close (signal already availability-lagged t+1 in the v0 panel); costs base case half_spread_bps=2, commission_bps=1, borrow_bps_pa=50; three cells only: base, doubled-costs, next_open-execution. One honest pass; failure per the gate's termination rule (`reports/promotion_gate_c11.md`) is final. | registration only — no statistic; graded against the ceiling at evaluation time (N=56 → 2.57) |
| 2026-07-13 | **gate2 cell 1/3 (base)**: C15 decile L/S per the registration above — close execution, costs 2+1 bp/side + 50 bp p.a. borrow, 2011-12-30 → 2026-04-02 (172 executed rebalances, last 2026-03-31) | net Sharpe **−0.00** (gross 0.27), ann net −0.00% (gross +1.74%), maxDD −28.5%, one-way turnover 20.7x/yr, break-even 4.21 bp/side; monthly-net HAC t = −0.01 (lags 1, 173 months) vs ceiling 2.57; FF5+MOM alpha −1.55%/yr (HAC t −0.90), mkt beta −0.006, R² 0.002; legs gross: long +2.42 / short −2.17 (report: `reports/gate2_c15_tradeable.md`) |
| 2026-07-13 | **gate2 cell 2/3 (doubled costs)**: same book, all cost components x2 | net Sharpe −0.27, ann net −1.75%, maxDD −37.5% → gate criterion 4 FAIL (negative under doubling) |
| 2026-07-13 | **gate2 cell 3/3 (next_open execution)**: base costs, trade at next day's open | net Sharpe −0.00, ann net −0.01%, break-even 4.18 bp/side — execution timing immaterial; the close-execution result is not a same-close artifact |
| 2026-07-13 | **Gate-2 post-run implementation audit (not a new look)**: `run_ls_backtest` forward-filled ticker signals across PIT membership exits and target formation checked price availability but not explicit membership. The engine now accepts daily eligibility, fails closed on missing ticker-dates, and resets carry across exit/re-entry spells; regression tests cover both cases. | The three registered numerical cells remain in the ledger but are not exact PIT estimates. They already failed the frozen economics gates, so C15's live path remains closed; no bug-fix rerun may rescue it. |

Hyperparameter variants (e.g. C6 MATCH_THRESH, chunk size) get ledger
lines too when swept.

| 2026-07-13 | **C17 registration (pre-computation; not a look)**: `sue_pead` — SUE = (EPS_q − EPS_{q−4})/std(last 8 seasonal diffs, min 6); diluted as-filed first-filed EPS from `xbrl_facts.parquet`; Q4 = annual − 3 quarters where no standalone Q4; availability = earliest of Item-2.02 8-K acceptance in (period_end, filed] else statement filed date; v0 t+1 at merge. Evaluation pinned: 1 primary look (v0 judge, sn incremental over BASELINE, 21d target) + 4 decay looks (`ic_decay` @ 10/21/42/63d). Five looks total when run. | registered |
| 2026-07-13 | **C17 std-window refinement (build-time pin, pre-evaluation; not a look)**: the registration's "std(last 8 seasonal diffs)" does not state whether the current diff sits in its own null; pinned as EXCLUDING the current one — std over the trailing 8 diffs before q (min 6, ddof=1, positional over the firm's observed quarters), so the scored surprise never overlaps its own baseline (consistent with C15's non-overlapping-null design). | pinned before any evaluation |
| 2026-07-13 | **C16 registration (pre-computation; not a look)**: `insider_cluster_buy` — ≥2 distinct natural-person officers/directors, code-P open-market buys, trailing 21d; size = Σ$ / 20d ADV; entry next close after 2nd filing; hold 63td. Person-vs-entity de-clustering required (joint filings repeat rows per co-owner CIK). Event breadth ~50–120/yr → XS-IC judge cannot score it; evaluation protocol (calendar-time event portfolio vs matched baseline) to be pinned here BEFORE any computation. | registered (definition only) |
| 2026-07-13 | **C18 registration (pre-computation; not a look)**: `nt_filing_veto` — first/rare Form 12b-25 as long-book exclusion flag, not a standalone short. NT forms not yet fetched (submissions-API pass). Evaluation protocol (exclusion delta on a baseline book) to be pinned BEFORE computation. | registered (definition only) |
| 2026-07-13 | **C17 look 1/5 (primary)**: sue_pead vs BASELINE, sn, v0 (pit=T lag=1 grid=daily, fresh panel). Build facts: 489/497 tickers with diluted EPS (25,569 first-filed quarterly obs; 28,808 rows incl. 11.2% derived Q4; 93.8% via 8-K 2.02) | sn incr t=+2.09 (IC +0.0090, ICIR +0.16), sn raw t=+1.08 (IC +0.0071), 168m, xs 370, cov 90.4%, ortho 0.923 (largest control corr mom_252_21 +0.22) — clears the pre-registered 1.5 candidate bar; below the ledger ceiling (~2.57) |
| 2026-07-13 | C17 look 2/5 (decay 10d): `ic_decay` on the monthly v0 panel (raw standalone IC by construction) | IC +0.0050, HAC t=+0.55 (lags 6, 176m) |
| 2026-07-13 | C17 look 3/5 (decay 21d): `ic_decay` on the monthly v0 panel | IC +0.0086, HAC t=+1.30 (lags 1, 176m) — the peak horizon, matching the 21d target |
| 2026-07-13 | C17 look 4/5 (decay 42d): `ic_decay` on the monthly v0 panel | IC +0.0042, HAC t=+0.41 (lags 4, 174m) |
| 2026-07-13 | C17 look 5/5 (decay 63d): `ic_decay` on the monthly v0 panel | IC +0.0002, HAC t=+0.01 (lags 4, 173m) — the drift is concentrated at ~1 month and gone by a quarter |
| 2026-07-13 | **filing_events items column (infrastructure; not a look)**: build filing_items.parquet — per-8-K item-code sets from line-anchored "Item #.##" headers, C17 ITEM_202_RE token generalized (scripts/build_filing_items.py; filing_acceptance.parquet untouched) | coverage 99.51% (100,121/100,618; zero-item files = table-mangled/glued headers and body-only mentions, not amendments); 2.02 reconciled with C17: anywhere-search 30,408 matches scan_8k_202 exactly; header-anchored 30,341 = strict subset, 67 mid-line-only mentions excluded |
| 2026-07-13 | **B7–B9 PIT fundamentals controls added (definitions in FACTORS.md; not a look)**: B7 `log_mktcap_pit` (PIT shares × close, BRK-B excluded), B8 `book_to_market_pit`, B9 `earnings_yield_pit` (as-filed first-filed XBRL; availability = the completing filing's filed date, TTM = max of the 4 constituents'; v0 t+1 at the merge; builder `signals/fundamentals_pit.py` → fundamentals_pit.parquet). No promotion testing, no evaluation looks. | BASELINE for the judge UNCHANGED (comparability with all prior looks preserved); an extended-controls set becomes available as `--accepted EXTENDED` |
| 2026-07-13 | **C19 registration (pre-computation; not a look)**: `cust_mom_pc` — the pre-registered C10-family confirmation on regulation-disclosed edges. Construction: EQUAL-weight mean of principal customers' 21d momentum; edges = `principal_customers.parquet` rows with evidence_verified AND tradeable (panel-resolvable), active in [valid_from, valid_to] (open windows stay active); daily as-of grid, computation-date stamps, v0 t+1 at merge. No confidence weighting, revenue_pct NOT used as weight (one construction, zero variants). Evaluation pinned: 3 looks — primary v0 judge (sn incremental over BASELINE) + ic_decay at {10, 21}d (10d = the original C10 claim's horizon). Success criterion pinned NOW: the family revives as candidate iff primary sn incr t ≥ +1.5 OR 10d HAC t ≥ +2.0 (original sign); anything less closes the customer-momentum family permanently. Prerequisite: backfill the 36 credit-blocked filings (WMB→YUM) so the one-shot runs on the complete corpus. Expected thin coverage (~43 panel customers / 78 suppliers): the confirmation tests sign+magnitude at pinned horizons, not ceiling clearance. | registered |
