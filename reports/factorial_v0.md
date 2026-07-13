# 2³ convention factorial — PIT × t+1 × grid attribution (v0 freeze input)

Run 2026-07-13, pre-committed in `plan_batch1_infra.md` (binding decisions
1–2). This is ATTRIBUTION of how three evaluation conventions move each
flagship factor's headline statistic — not selection. No cell was re-run,
no conclusion about which factor is good is drawn here, and every cell is
one ledgered look (19 rows appended to `factor_preregistration.md`, same
date, tag `factorial`). Ledger N grows by 19; per binding decision 1,
E[max|null] must be re-derived against the grown N before any candidate is
called surviving — this report deliberately contains no survival language.

## Protocol

- **Panels built once** (9 parquets, session scratchpad):
  8 factorial corners — monthly-signal and daily-signal grids × pit {F,T} ×
  lag {0,1} — plus one sensitivity panel (pit=T, lag=1, daily, controls
  lagged one trading day via the new `lag_controls` builder switch).
  Non-PIT panels: 1,774,402 rows / 499 tickers; PIT: 1,453,102 / 494
  (18.1% of rows removed). The grid axis changes which CACHE feeds the
  candidate column (and whether C15 exists); the judge samples every panel
  to monthly rows internally, so "daily grid" means daily-updated signal
  values evaluated at monthly rows, not daily-frequency evaluation.
- **One judge invocation per cell** (19 total):
  `factor_orthogonality.py evaluate --panel <cell> --candidate <X>
  --accepted BASELINE --sector-neutral`.
- **What "raw" means here.** `evaluate()` reports raw and incremental IC
  from ONE invocation; `--sector-neutral` demeans all rank-z series
  (candidate, controls, target) within sector for BOTH numbers — it is a
  mode switch, not an additional output. Each cell was run exactly once, in
  sector-neutral mode (the registry's headline mode). The raw-t column
  below is therefore the sector-neutral unresidualized t; plain
  non-neutralized numbers were not computed anywhere in this factorial
  (running them would have doubled the look count to 38).
- **Coverage** = % of monthly-sampled rows (the judge's
  last-row-per-ticker-month rule) with a non-null candidate, computed from
  the saved panels without reading returns (not a look).
- **Sensitivity-panel integrity** (artifact-verified, no returns touched):
  `d_p1_l1_cl` differs from `d_p1_l1` only in the 6 control columns —
  candidates bit-identical; controls equal the pre-mask one-trading-day
  shift exactly (0 mismatches over 1,453,102 rows × 6 columns). The shift
  runs before the PIT mask, so control lookbacks never span membership gaps
  (same ordering reason as the review-F1 fix).

## Axis degeneracies (encoded, then verified on the built panels)

1. **Lag is inert for grid-signal merges on the monthly grid.** The
   availability lag applies to filing-date merges on every grid but to
   grid-signal merges only under `daily_signals` (a +1d shift on a
   month-end-stamped series is the stale variant, not t+1). Verified:
   C10-monthly and C11 candidate columns bit-identical across lag on both
   pit strata. C10-monthly therefore contributes 2 cells (lag 0 ≡ lag 1).
2. **C11 is additionally grid-inert.** Its merge is deliberately never
   lagged and it loads the monthly cache regardless of grid. Verified:
   C11 column identical between `m_p1_l0` and `d_p1_l1`. C11's only live
   axis is PIT → 2 cells.
3. **C1 is grid-inert.** Verified: C1's full evaluation input set
   (candidate, target, sector, all 6 controls) bit-identical between
   `m_p1_l1` and `d_p1_l1` over all 1,453,102 rows — so C1's monthly-panel
   cells ARE its daily-grid values, including the v0 cell.
4. **C15 exists only on the daily grid** (4 cells).

## The 19 cells

Headline = sector-neutral incremental-IC t over the 6-factor price/volume
baseline. Raw t = same invocation's sector-neutral unresidualized t.
Grid `m (=d)` / lag `0 (=1)` mark verified-inert axes.

| cell | factor | pit | lag | grid | ctl lag | sn incr t | sn incr IC | sn raw t | months | avg xs | cand cov % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1_m_p0_l0 | C1 | F | 0 | m (=d) | F | +1.48 | +0.0066 | +2.11 | 168 | 451 | 89.8 |
| C1_m_p0_l1 | C1 | F | 1 | m (=d) | F | +1.47 | +0.0065 | +2.08 | 168 | 451 | 89.7 |
| C1_m_p1_l0 | C1 | T | 0 | m (=d) | F | +0.29 | +0.0014 | +0.74 | 168 | 382 | 92.5 |
| C1_m_p1_l1 | C1 | T | 1 | m (=d) | F | **+0.24** | +0.0011 | +0.70 | 168 | 382 | 92.5 |
| C10_m_p0_l0 | C10 | F | 0 (=1) | m | F | +1.40 | +0.0093 | +1.68 | 165 | 178 | 34.7 |
| C10_m_p1_l0 | C10 | T | 0 (=1) | m | F | +1.05 | +0.0076 | +1.01 | 165 | 160 | 38.1 |
| C10_d_p0_l0 | C10 | F | 0 | d | F | +1.44 | +0.0097 | +1.57 | 166 | 177 | 34.8 |
| C10_d_p0_l1 | C10 | F | 1 | d | F | +1.05 | +0.0070 | +1.23 | 165 | 178 | 34.7 |
| C10_d_p1_l0 | C10 | T | 0 | d | F | +0.91 | +0.0064 | +1.02 | 166 | 160 | 38.2 |
| C10_d_p1_l1 | C10 | T | 1 | d | F | **+0.21** | +0.0015 | +0.33 | 165 | 161 | 38.1 |
| C11_m_p0_l0 | C11 | F | 0 (=1) | m (=d) | F | −2.81 | −0.0110 | −3.20 | 168 | 458 | 96.0 |
| C11_m_p1_l0 | C11 | T | 0 (=1) | m (=d) | F | **−2.00** | −0.0089 | −2.42 | 168 | 386 | 97.4 |
| C15_d_p0_l0 | C15 | F | 0 | d | F | −3.29 | −0.0138 | −3.47 | 168 | 458 | 93.1 |
| C15_d_p0_l1 | C15 | F | 1 | d | F | −3.67 | −0.0156 | −3.72 | 168 | 458 | 93.1 |
| C15_d_p1_l0 | C15 | T | 0 | d | F | −1.85 | −0.0090 | −2.35 | 168 | 386 | 95.0 |
| C15_d_p1_l1 | C15 | T | 1 | d | F | **−2.29** | −0.0111 | −2.63 | 168 | 386 | 95.0 |
| C1_d_p1_l1_cl | C1 | T | 1 | d | T | +0.10 | +0.0005 | +0.70 | 168 | 382 | 92.5 |
| C10_d_p1_l1_cl | C10 | T | 1 | d | T | −0.05 | −0.0003 | +0.34 | 165 | 161 | 38.1 |
| C15_d_p1_l1_cl | C15 | T | 1 | d | T | −2.33 | −0.0112 | −2.63 | 168 | 386 | 95.0 |

Bold = v0-convention cells (see below). All 19 runs returned cleanly; no
cell failed the MIN_XS/MIN_MONTHS gates. Coverage RISES 2–3.5 pts under
PIT for every factor (non-members are sparser in filings and graph edges
than members) — no coverage collapse anywhere. C10's 166-vs-165 month
counts: the daily cache adds one thin early month over the MIN_XS=20 gate
at lag 0 (plus 133 daily-only sampled ticker-months).

## Main effects per factor (descriptive; mean headline-t change when the axis flips on; no significance claims about these differences)

**C1 `cosine_similarity`** (monthly 2×2; grid exactly inert, verified)
- PIT: **−1.21** (+1.48/+1.47 → +0.29/+0.24). The dominant axis by far.
- t+1 lag: **−0.03** (≈ nothing: the filing-date merge moves by one day,
  which monthly sampling almost never lands on).
- Grid: 0 by construction (verified bit-identical inputs).
- Controls lagged (single v0-corner sensitivity): −0.14 (+0.24 → +0.10).

**C10 `spillover_cust_mom`** (daily 2×2 is the full-resolution stratum)
- PIT: **−0.69** (mean +1.245 → +0.56). On the monthly grid: −0.35
  (+1.40 → +1.05).
- t+1 lag: **−0.55** (mean +1.175 → +0.63).
- pit × lag (descriptive): the lag effect is −0.39 under pit=F and −0.70
  under pit=T — the two conventions compound; the v0 corner is +0.21.
- Grid at lag 0: +0.04 (pit=F) / −0.14 (pit=T) — small. Mechanism note:
  the monthly cache stamps CALENDAR month-ends (50/169 on weekends), so at
  ~35% of sampled ticker-months the as-of merge carries the PRIOR month's
  value; the daily cache is fresh at the sampled row. Month-end grid values
  themselves are pinned identical by the builder test — the difference is
  merge staleness, exactly the granularity the grid axis exists to expose.
- Controls lagged: −0.26 (+0.21 → −0.05; sign crosses zero, both values
  noise-sized).

**C11 `evt8k_freq_z`** (PIT is the only live axis; lag and grid verified inert)
- PIT: **+0.81 signed** (−2.81 → −2.00), i.e. |t| shrinks 2.81 → 2.00.

**C15 `evt8k_freq_z_d`** (daily 2×2; exists only on this grid)
- PIT: **+1.41 signed** (mean −3.48 → −2.07): attenuation toward zero, the
  dominant axis.
- t+1 lag: **−0.41 signed** (mean −2.57 → −2.98): |t| INCREASES under t+1
  in both PIT strata (−3.29 → −3.67 and −1.85 → −2.29). Directionally
  consistent with the C11 review's observation that same-close attachments
  dilute (excluding last-day filings strengthened C11 to −2.94); it is NOT
  the fresh/stale-month split, and the near-coincidence of −3.67 with that
  review's fresh-attach diagnostic (−3.68) is numerical accident — they are
  different quantities.
- pit × lag: −0.38 vs −0.44 — near-parallel, minimal interaction.
- Controls lagged: −0.04 (−2.29 → −2.33), negligible.

Sensitivity-axis summary: lagging the 6 in-panel controls to the same t+1
standard moves the event factor C15 by −0.04 and moves the two
already-noise-sized momentum-adjacent candidates toward/through zero
(C1 −0.14, C10 −0.26) — the candidate-vs-control comparison for weak
momentum-family candidates is sensitive to control timing at exactly the
size of their remaining t.

## Flags

**(a) Degeneracies.** Two, both artifact-verified (bit-identical candidate
columns, share 1.000000): monthly-grid lag inertness for C10/C11, and
C11's grid inertness (its merge is never lagged BY DESIGN — a +1d shift on
its month-end stamps would produce the already-tested stale variant, not
t+1). The factorial therefore has 19 informative cells, not 8 × 4.

**(b) C15 vs C11 is not a grid effect.** C15 bundles three definitional
changes (daily grid; trailing-21-trading-day window vs calendar month;
lagged non-overlapping baseline vs an inclusive rolling 24-month one — the
last mechanically inflates |z|). That is why C15 is its own registry ID.
The C15−C11 gap at pit=F (−3.29 vs −2.81 at lag 0) must not be read as
"the daily grid strengthens C11".

**(c) v0 cells (pit=T, lag=1, daily grid) — the platform's headline
numbers after the v0 freeze**, printed with avg cross-section per the PIT
language rule (forward-bias removal only, survivor bias remains: departed-
name gap 194 names in 2011 → 23 in 2025; no PIT delta is a "correction"):

| factor | v0 headline (sn incr t) | sn incr IC | months | avg xs |
|---|---|---|---|---|
| C1 | +0.24 | +0.0011 | 168 | 382 |
| C10 | +0.21 | +0.0015 | 165 | 161 |
| C11 (pit=T; lag/grid inert for it) | −2.00 | −0.0089 | 168 | 386 |
| C15 | −2.29 | −0.0111 | 168 | 386 |

C1's v0 number comes from the `m_p1_l1` cell, verified bit-identical in
evaluation inputs to `d_p1_l1`.

**(d) Cells exceeding prior headline |t|.** C15_d_p0_l0 (−3.29) and
C15_d_p0_l1 (−3.67) exceed the largest prior registry headline (C11
monthly −2.81). Both are pit=F cells; under PIT the same construction
reads −1.85 / −2.29. Attribution only: the excess over C11 sits entirely
in the non-PIT corners, which the v0 convention excludes. The ledger N is
now 19 larger; no E[max|null] comparison is made here.

## Reproduction anchors

- C11 pit=F cell −2.81 = the registry headline exactly.
- C1 pit=F cell +1.48 matches the ledger's clean C1 sector-neutral 1.5.
- C10 monthly pit=F +1.40 vs the pre-dedup-fix +1.54: both C10 caches were
  rebuilt 2026-07-13 after the edge-dedup determinism fix; the ledger
  predicted tie-edge-sized deltas, which this is.

## Wall clock

- Panel builds: 3.1–3.7 s (non-PIT), 6.1–6.6 s (PIT; membership load adds
  ~3 s); 45.5 s for all 9 panels.
- Judge evaluations: 1.6–1.7 s per cell; ~31 s for all 19.
- Full factorial including verification checks: well under the plan's
  "minutes" budget [E7].
