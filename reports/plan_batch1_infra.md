# Platform v0 — Batch 1 Infrastructure Plan (v2)

v1 was reviewed by three independent Opus reviewers (statistics, engineering,
residual-bias lenses); all three returned approve-with-changes. v2 incorporates
every required change. Review evidence is cited inline as [S#], [E#], [B#].

Four items: PIT membership, t+1 availability, diagnostics library, daily
signal grid. Execution order: **3 → 1 → (2+4 jointly) → factorial
re-evaluation.**

## Binding decisions from review (apply across all items)

1. **Convention freeze [B3, S4].** v0 = (PIT on, t+1 on, daily grid) is
   pre-committed as the sole externally-quoted convention, declared here
   before any run. Every other (factor × convention) cell is internal
   robustness only. Every cell computed — including the factorial below —
   is added to the ledger N, and E[max|null] is re-derived against the grown
   N before any candidate is called surviving.
2. **Factorial attribution, not one-at-a-time [S2].** Deltas are attributed
   via the full 2^3 factorial (PIT × t+1 × grid = 8 corners) on the flagship
   factors (C1, C10, C11), reporting main effects and interaction terms. OAT
   is invalid here: the plan itself documents a sign-flipping t+1 × grid
   interaction for C11. Cost is minutes (panel build 3.1 s, evaluate 4.5 s
   [E7]).
3. **PIT is a forward-bias fix only [B1, S1].** Measured: the 2011 index has
   497 names, the panel overlaps 285 (43% unrecoverable — departed names have
   no price/filing data). Language rules: no PIT delta is called a
   "correction"; every PIT number is printed with its avg cross-section and
   the departed-name gap for that year. Survivor bias remains open (task #9).
4. **The daily C11 construction is a NEW variant, not a port [S4, B2].** It
   bundles three changes (grid, window type, live-window baseline exclusion —
   the last mechanically inflates |z|). It gets a new registry ID and its own
   ledger lines (one per sub-change, measured separately in the factorial);
   C11-monthly's −2.81 remains the registry headline until an out-of-design
   confirmation. The daily number is additionally reported on data not used
   to discover the fresh-month effect (the 2020–26 half and the thin-coverage
   names) to blunt the forking-paths concern.

---

## Item 1 — PIT S&P 500 membership in the panel

**Objective.** Evaluate factors only on names that were index members at each
date. Forward-bias removal only (see binding decision 3).

**Data facts (measured [E]).** CSV: 2,705 snapshots 1996-01-02→2026-01-14,
dotted symbols (`BRK.B`), cross-section 496–507 post-2011; a change-log after
2018 (median gap 2 d, 21 gaps > 35 d, max 122 d — legitimate no-change
periods, NOT missing data [E2, B4]). Panel: dashed symbols, prices to
2026-04-02 (2.5 months past the last snapshot [E8]).

**Design.**
- Mask inside `build_feature_panel`, flag `pit_universe: bool = True`.
- Membership calendar: as-of forward-fill of snapshots (correct for a
  change-log). Validation: dates strictly increasing; cross-section within
  [490, 515] (verified non-firing on the real file [E2]). The v1 ">35-day
  gap → fail" rule is deleted — it fires 21× on real data and inverts the
  risk. Deletion-lag exposure (a departed name carried up to 122 d) is
  cross-checked: flag any name with panel prices after its as-of removal or
  missing prices while still "in" [B4].
- Membership carry-forward is bounded: after the last snapshot (2026-01-14),
  membership is held ≤ 60 days; panel dates beyond that are dropped from PIT
  mode unless the CSV is refreshed [E8].
- **Symbol drift is the load-bearing correctness problem [E1].** The CSV is
  symbol-as-of-date (META appears only from 2022-06-09; before that it is FB).
  A missing rename silently deletes a mega-cap for years. The v1 90% coverage
  gate is deleted — measured raw match is 64–83% pre-2021 for structural
  reasons (legitimate later joiners), so it can neither pass nor detect a
  single missing rename. Replacement: **per-ticker rename-completeness
  check** — for every panel ticker whose price history predates its first
  membership match, classify individually: IPO/later-joiner (correctly
  excluded) vs rename (must be in the map). The rename map is exhaustive,
  versioned, and ledgered as an artifact [B5]; zero silent drops is a hard
  test, not a soft alert.

**Acceptance.**
- Exclusion test: TSLA (added 2020-12-21; 9.5 years of pre-join price rows)
  has no PIT rows before its join date [E6].
- **False-drop test (merge gate): META has PIT panel rows in 2016.** This
  test fails before the rename map exists and passes after [E6].
- Gap-handling test on synthetic snapshots containing a 100-day gap.
- Per-year report: avg cross-section, departed-name gap, unmatched-symbol
  list (adjudicated, none silent).

**Effort.** 2–3 days (rename triage is ~150 names in early years, each
classified [E3]); re-evaluation itself is seconds.

---

## Item 2 — t+1 availability convention

**Objective.** A signal stamped day t becomes usable at the first close
after t.

**Design.**
- `availability_lag_days: int` is an explicit per-call-site argument,
  **default 0**. Set to 1 only on the three filing-date merges (text factors).
  The three grid-date merges (spillover, customer momentum, C11) do NOT get
  the merge-layer lag — their availability is handled inside their builders,
  which own the as-of semantics. One owner per signal; C11 must not be
  double-lagged [E5].
- C10's inputs are neighbor prices at close t; its builder applies the +1 so
  its price inputs and the target do not share close(t) [B7]. Pinned by unit
  test.
- +1 calendar day then backward-merge = next-trading-close semantics
  (holiday-safe, verified [S-verdict]); conservative by up to one trading day
  intraday — documented.
- **Anchor asymmetry is documented, not hidden [S5]:** in the incremental-IC
  regression, baselines sit at close(t) while lagged candidates sit at
  close(t−1); shared information is awarded to the baselines, biasing
  candidate incremental IC downward (worst for fast signals). The factorial
  includes one sensitivity cell with controls lagged +1 for C11 to bound the
  effect; the ledger states the direction wherever incremental numbers are
  quoted.

**Acceptance.** Unit tests: signal dated the sampled day does not attach;
the day before does; Friday→Monday pinned; C11 not double-lagged (builder
lag on + merge lag off).

**Effort.** ~0.5 day + re-evaluation.

---

## Item 3 — Diagnostics library `ic_tools`

**Objective.** Promote the scratchpad analyses into tested, importable
functions; kill the caught bug classes (split-boundary drops, overlap-naive
t-stats, slow-factor inflation).

**API** (`src/alpha_graph/eval/ic_tools.py`):
- `monthly_ic(panel_m, factor, target, min_names=20) -> Series`
- `ic_summary(ic, hac_lags=None, n_trials=None) -> dict` — mean, std,
  t_naive, t_hac, icir, n_months, hit_rate, and `deflated_t` when `n_trials`
  is given. **DSR/deflated-t is a first-class top-level function**
  (`deflated_t(t, n_trials, T)`), not buried [S7].
- `hac_tstat(x, lags) -> float` (Newey-West/Bartlett).
- `default_lags(horizon_days, sampling_interval_days) -> int` =
  `ceil(horizon/interval)`, **with a persistence floor**: lags are raised to
  the first lag where the measured IC autocorrelation crosses zero (capped).
  The rule is sampling-aware because Item 4 introduces weekly sampling, where
  a 21-day horizon overlaps ~4 observations [S3].
- **Guardrail: naive t is not emitted when sampling interval < holding
  horizon** — HAC is forced, in `monthly_ic`/`quantile_ls` alike [B6].
- `ic_decay(panel, factor, prices, horizons=(5,10,21,42,63,126))` — per
  horizon: mean IC, HAC t (per-horizon lags).
- `split_halves(ic, boundary)` — first = index ≤ boundary, second = index >
  boundary; regression test constructed so an off-by-one boundary changes the
  answer.
- `quantile_ls(...) -> dict` — adds **break-even one-way cost** (bps at which
  L/S Sharpe = 0) next to turnover, per rebalance frequency [S7].
- `effective_n(factor_matrix)` — Li-Ji/Nyholt on pooled rank correlations.
  **Labeled: measures factor-column redundancy only. The significance
  ceiling `emax_null` is fed the ledger look-count N, never this m_eff**
  [S7].
- `rank_autocorr(panel_m, factor)` — mean consecutive-month rank Spearman;
  `warn_slow` threshold **0.65** [S3]; wired into `default_lags`'
  persistence floor, not a passive warning.
- `sector_neutralize(sub, cols, sector_col)`.
- Auto-emitted sub-period/multi-horizon outputs are tagged `look=True` so
  they are countable into N rather than silently free [S7].

**Refactor gate [E4].** `factor_orthogonality.py` imports the *moved*
functions unchanged (literal bit-identity); statsmodels/new-HAC code lives
only in new paths and is cross-checked against statsmodels in tests, never
gating the legacy numbers. Automated pre/post equivalence test on a fixed
panel slice replaces the manual "printed precision" check.

**Effort.** ~1.5 days.

---

## Item 4 — Daily as-of signal grid

**Objective.** Signals defined every trading day. Root fix for C11's
fresh/stale mix; makes t+1 measurable; unlocks weekly-rebalance research.

**Design.**
- **New variant discipline per binding decision 4.** Daily construction:
  `count_21(t)` = filings in (t−20…t] trading days; baseline mean/std over
  `count_21` on (t−524…t−21]; std=0 → NaN. Registered with a new ID; the
  grid change, window-type change, and live-window exclusion are ledgered
  and measured as three separate lines in the factorial.
- Trading-day calendar = the price panel's per-ticker date index; non-trading
  filings roll to the next trading day's window.
- **Graph series go daily too.** The v1 weekly compromise rested on a wrong
  cost model: `build_graph` is 0.04 s/call → daily ≈ 2.7 min once the market
  frame is loaded once and sliced (the v1 "30 min" was an avoidable re-read
  of the 107 MB parquet per date [E7]). Daily grid removes the weekly
  overlap trap entirely [B6].
- Weekly *sampling* mode in the evaluator (for the C10 10-day-horizon work)
  ships only behind the Item-3 guardrail: HAC with
  `default_lags(horizon, 5)`, no naive t [S3, B6]. C10 is additionally
  evaluated at its claimed 10-day horizon via `ic_decay`, not only at the
  21-day target [S8].
- Storage: daily parquet per factor; C11 ≈ 1.9M rows — trivial [E7].

**Acceptance.**
- No value at date t uses a filing dated > t (planted-filing test, with the
  builder-level lag from Item 2).
- The new-variant daily number is reported (a) full sample, (b) 2020–26
  held-out half, (c) thin-coverage names — per binding decision 4; whatever
  it is, C11-monthly's −2.81 stays the registry headline pending
  out-of-design confirmation.
- First weekly-sampling measurements ledgered with HAC only.

**Effort.** ~1 day.

---

## Joint acceptance for v0

1. Factorial (2^3 × {C1, C10, C11}, plus the controls-lagged sensitivity
   cell for C11) run and ledgered; main effects and interactions reported;
   all cells counted into N; E[max|null] re-derived against the grown N.
2. All standing candidates re-evaluated under the frozen v0 convention with
   `ic_tools` summaries (HAC, decay, halves, turnover + break-even cost,
   rank-autocorr) attached.
3. FACTORS.md/README restated: v0 numbers labeled with avg_xs and the
   departed-name gap; superseded numbers kept as tombstones; no "correction"
   language for PIT deltas.
4. Tests green, including: META-2016 false-drop gate, TSLA exclusion,
   C11 no-double-lag, split-boundary regression, pre/post refactor
   equivalence, statsmodels HAC cross-check.
5. Membership CSV refreshed or its carry-forward bound respected before the
   v0 numbers are frozen.
