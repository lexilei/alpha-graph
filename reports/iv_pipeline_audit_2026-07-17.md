# Adversarial audit — options IV pipeline (2026-07-17)

Read-only review by an independent agent of `src/alpha_graph/options/bs.py`,
`options/iv.py`, `scripts/validate_iv_pipeline_spy.py`,
`tests/test_options_iv.py`. Every claim demonstrated on the actual SPY
clean store / real single-name data; all validation-CSV headline numbers
re-derived exactly.

## FINDINGS (ranked)

**F1 — HIGH, factor-critical: carry double-count.** `parity_spot` returns
the FORWARD F ≈ S·e^{(r−q)T} (confirmed +0.27% on SPY at 37 DTE); feeding
it into spot-BS at rate=0.04 discounts the strike against an
already-carried underlying → `iv_call − iv_put ≈ −r√T/φ ≈ −3.2 vol pt`
at 37 DTE. Demonstrated: synthetic flat vol −0.0322; real SPY all-days
median −0.0322; real AMZN day −0.0316; rate=0 → exactly 0. The common
part cancels cross-sectionally, but two residuals do not: (a) the wedge
scales with √T — ~0.8 vol pt of dispersion across the 28-46 DTE band from
each name's expiration calendar; (b) a fake OTM put skew (+0.7 vol pt at
37 DTE, vol-level-correlated — 0.56 low-vol vs 1.12 high-vol). These land
exactly on the registered I2 (cp spread) and I1 (skew) constructions.

**F2 — HIGH, validation blind spot.** The SPY gate compares only the
call/put AVERAGE — the one combination where the wedge cancels. A green
gate certified the ATM-IV level, not the leg-level quantities the factors
consume.

**F3 — MED: tie-break comment claims parity with vol_smile but is opposite
on the 120 divergence days** (all exact DTE ties; ours later, vol_smile
earlier). Suggested flipping to earlier. **REFUTED on disposition — see
below.**

**F4 — MED: no crossed/locked-quote filter** (`ask <= bid` kept; vol_smile
drops them + a (0,3) IV band). Harmless on SPY (0 selected-strike hits,
1 day changed); structurally exposed on wide-spread single names — a
crossed mid enters `parity_spot` (spot = K+C−P) directly.

**F5 — MED: one-legged `.iv` biased by ~half the wedge** (±1.6 vol pt)
on failed-inversion days; collapses once F1 is fixed.

## VERIFICATION (by the auditor)

Verbatim BS port CONFIRMED — all four ported functions AST-extracted and
md5-identical to vol_smile's `black_scholes.py`. Validation CSV re-derived
exactly (2520/2520; same-sel n=793 median 0.00035; overall 0.00224 /
0.00662; spot 0.22818%). RIGHT_MAP verified against both stores' actual
`right` encodings. `parity_spot` min-|C−P| robust on 2522 SPY days.

## DISPOSITION (same day, commit `be2eddb`)

- **F1/F2/F5 fixed**: leg inversion is now exact Black-76 — each mid
  undiscounted by e^{rT}, inverted at rate 0 against the parity forward;
  bs.py untouched. Validation gains iv_call/iv_put columns + a wedge
  diagnostic; flat-vol no-wedge synthetic test added. Post-fix:
  **iv_call − iv_put median +0.00000 (p5/p95 ±0.00006)** — wedge gone;
  same-selection median 0.00083 (n=793, still ≤ the 0.001 gate; the shift
  from 0.00035 is Black-76-with-dividends vs the audited engine's
  q=0 spot-BS), overall median improved 0.00224 → 0.00209; PASS.
- **F4 fixed**: `ask <= bid` dropped in `paired_mids` + (0,3) leg band;
  synthetic chains in tests now carry real spreads (a zero-spread chain
  would be filtered), crossed-poisoning test added.
- **F3 refuted, comment corrected instead**: flipping to earlier dropped
  same-expiry agreement 95.2% → 85.3%. The audit inspected only the
  later-pin's 120 divergence days — by construction the ties vol_smile
  resolved earlier — and missed the ~250 tie days it resolved later
  (row-order resolution, ~3/4 later). Selection effect; the later pin is
  the majority behavior and stays. Lesson recorded: an audit's fix
  direction gets the same artifact-verification bar as the code it
  audits.
