# Gate-2 tradeable backtest: C15 decile long-short (2026-07-13)

The registered tradeable-backtest pass for C15
(`evt8k_freq_z_d`, daily abnormal 8-K filing frequency), graded against the
frozen promotion gate `reports/promotion_gate_c11.md`. The design below was
fixed before any result existed; nothing was tuned, and per the gate's
termination rule the outcome is final.

**Verdict: FAIL — the live path terminates.** Gross spread ≈ +1.74%/yr is
consumed almost exactly by base-case costs (net Sharpe −0.00); doubled costs
are decisively negative; break-even is 4.2 bp/side against a required
6.0 bp/side. The C15 research-level result (v0 sector-neutral incremental
t = −2.29, itself below the ledger ceiling) stands as what it is: a real but
untradeably small predictive effect.

## Post-run implementation audit

An implementation review later on 2026-07-13 found that the engine forward-
filled each ticker's last signal indefinitely and checked only price presence
when forming a target. A stock removed from the PIT universe could therefore
remain selectable with a stale pre-removal score. The engine now accepts an
explicit daily eligibility frame and resets signal carry across ineligible
spells; regression tests cover both removal and re-entry.

The numerical tables below are retained as the immutable record of the
registered run, but they are not an exact PIT performance estimate and must
not be cited as one. The run already failed the frozen economics gates by a
wide margin, so its live path remains closed. The bug fix does not authorize a
rescue rerun; any later full-universe replication is a research data-integrity
exercise only.

Three further review annotations (2026-07-13, five-lens review; the recorded
tables are unchanged):

1. **Reruns at HEAD differ from the recorded tables.** The script now passes
   the eligibility mask, so a rerun gives the corrected-engine numbers (base
   gross +1.91%/yr, net Sharpe +0.02, break-even 4.54 bp/side, monthly HAC
   t +0.08 — criteria 2/4/5 still fail decisively; the old stale-score bug
   *depressed* the recorded numbers). The recorded tables reproduce only at
   pinned commit `3a86cad`.
2. **The FF alpha row (−1.55%/yr, t −0.90) is a convention artifact.** It was
   computed under `excess=False` (rf subtracted from a self-financing
   dollar-neutral spread); under the correct `excess=True` convention added
   later the same day, alpha is +0.06%/yr (t +0.03) — i.e. ≈ −rf was the
   entire "negative alpha." No gate criterion read this number; criterion 3
   used the monthly net HAC t.
3. **Wording correction:** the phrase "nets to zero by construction of the
   break-even" below is wrong as written — the base case used 3 bp/side, not
   the 4.21 bp break-even. Net ≈ 0 because trading (1.24%/yr) + borrow
   (0.50%/yr) happens to ≈ gross (1.74%/yr): a coincidence of magnitudes,
   not a construction.

## Registration (echo)

Appended to `reports/factor_preregistration.md` before computation:

> Gate-2 tradeable backtest, C15 decile L/S, registered pre-run: long
> lowest-z decile, short highest-z decile (sign per C15's negative IC);
> monthly rebalance at each month's last trading close using that day's
> available signal; execution at the rebalance close (signal already
> availability-lagged t+1 in the v0 panel); costs base case
> half_spread_bps=2, commission_bps=1, borrow_bps_pa=50; three cells only:
> base, doubled-costs, next_open-execution. One honest pass; failure per the
> gate's termination rule is final.

## Data and engine provenance

- **Signal**: v0 feature panel via `build_feature_panel()` pure defaults
  (PIT membership, t+1 availability lag, daily grid — the frozen v0
  conventions), column `evt8k_freq_z_d`, non-null rows only: 1,379,112 rows,
  494 tickers, 2011-12-02 → 2026-03-04. The panel's
  `dropna(fwd_return_21d)` trims the final ~21 trading days of the sample,
  so the signal series ends 2026-03-04; the last executed rebalance is
  **2026-03-31** (close cells), formed from the 2026-03-04 as-of signal
  (27 calendar days stale — one rebalance out of 172; the 2026-04-02
  month-end decision produced an identical target and traded nothing).
  P&L runs through 2026-04-02.
- **Prices**: `data/cache/market_data.parquet` (close, and open for the
  next_open cell), 499 tickers, 2011-04-06 → 2026-04-02.
- **Engine**: `src/alpha_graph/portfolio` — `backtest.py`, `report.py`,
  `__init__.py` at commit `d9a2f8c`; `construction.py`, `costs.py` at
  `fc2a14c`. Panel builder `ml_combiner.py` at `f120563`; `ff_attribution.py`
  at `ad95f13`; `ic_tools.py` at `a2f8225`. Run at HEAD `3a86cad` via
  `scripts/gate2_c15_backtest.py`.
- **Construction**: decile L/S (`n_quantiles=10`), `direction=-1` so the
  long leg holds the LOWEST-z decile (verified on a synthetic frame, and on
  the first real rebalance: long-leg mean z = −1.64, short-leg mean
  z = +2.11). Equal weight within leg, per-name cap 0.05 non-binding
  (~38 names/leg → 2.6%). 173 month-end decisions, 172 executed, 0 skipped
  for small cross-sections, 0 delisting liquidations.

## Results — the three registered cells

| metric | base (close) | doubled costs | next_open |
|---|---|---|---|
| net Sharpe (ann) | **−0.00** | −0.27 | −0.00 |
| gross Sharpe (ann) | 0.27 | 0.27 | 0.27 |
| ann return net | −0.00% | −1.75% | −0.01% |
| ann return gross | +1.74% | +1.74% | +1.73% |
| max drawdown (net) | −28.5% | −37.5% | −28.7% |
| one-way turnover (ann) | 20.7x | 20.7x | 20.7x |
| break-even cost (bp/side) | 4.21 | 4.21 | 4.18 |
| beta vs EW universe | −0.006 | −0.006 | −0.006 |
| long-leg gross total | +2.42 | +2.42 | +2.42 |
| short-leg gross total | −2.17 | −2.17 | −2.17 |
| top-10-name share of gross P&L | 155% | 155% | 156% |
| max sector share (Technology) | 121% | 121% | 112% |
| sample | 2011-12-30 → 2026-04-02 | same | 2012-01-03 → 2026-04-02 |

Monthly turnover ≈ 1.73 of a gross-2.0 book: ~86% of positions change decile
each month, as expected for a signal whose predictive content decays within
one month. Cost identity check: 2 sides x 20.7x/yr x 4.21 bp = 1.74%/yr =
the gross return — the base case nets to zero by construction of the
break-even.

Per-year net returns, base cell (compounded): 2011 −0.1% (Dec only),
2012 +0.1%, 2013 −6.9%, 2014 +4.0%, 2015 −5.1%, 2016 +1.3%, 2017 −2.6%,
2018 −4.3%, 2019 −0.5%, 2020 −7.0%, 2021 +3.0%, 2022 +4.8%, 2023 +7.9%,
2024 +5.7%, 2025 −3.0%, 2026 +1.1% (through 2026-04-02). Total net P&L ≤ 0,
so leave-one-year-out and year-concentration shares are undefined; the
name/sector shares quoted above are measured against a near-zero gross total
(+0.25 over 14.3 years) and are correspondingly unstable.

## Base-cell statistics

- **Monthly net HAC t** (gate criterion 3): monthly compounded net returns,
  173 months; HAC lags = 1 via `ic_tools.default_lags(21, 21, ic=series)`
  (the library's rule for a monthly non-overlapping series; the persistence
  floor did not raise it). **t = −0.01**, against the full-ledger ceiling at
  evaluation time `emax_null(2N)` = emax_null(112) = **2.57** (N = 56 per
  the 2026-07-13 ledger accounting, after the decay-sweep back-fill; the
  three gate2 cells themselves land after that count and would move the
  ceiling to 2.59 without changing any verdict).
- **FF5+MOM attribution** (net daily series, HAC lags 32,
  `alpha_graph.eval.ff_attribution`): alpha −1.55%/yr, **alpha HAC
  t = −0.90**, mkt_rf beta −0.006 (t −0.46), R² = 0.002, residual vol
  6.5%/yr. The book is genuinely dollar-neutral and factor-neutral — there
  is simply almost nothing left after costs to attribute.
- **Leg decomposition** (gate criterion 7): gross contributions long +2.42 /
  short −2.17 (arithmetic sums of daily leg contributions, units of initial
  NAV). Both legs are dominated by 14 years of market drift in opposite
  directions; the L/S signal content is their +0.25 spread. The short
  (high-z) leg loses money outright gross, so the strategy is NOT
  short-only-carried; nothing here supports a long-only reading either —
  the spread, not a leg, is the effect.

## Gate verdict (criteria from `reports/promotion_gate_c11.md`)

| # | criterion | measured | verdict |
|---|---|---|---|
| 1 | forward/paper P&L positive on unused data | accrues post-2026-04 | N/A (not yet accrued) |
| 2 | net Sharpe ≥ 0.8 after full costs | **−0.00** | **FAIL** |
| 3 | monthly-net HAC t clears ledger ceiling 2.57 | t = −0.01 | **FAIL** |
| 4 | non-negative with all costs + borrow doubled | −1.75%/yr (Sharpe −0.27) | **FAIL** |
| 5 | break-even ≥ 2x actual cost estimate: 2 x 3.0 = 6.0 bp/side | 4.21 bp/side (1.40x) | **FAIL** |
| 6 | no concentration (year > 60% / sector > 50% / top-10 > 50%) | year shares undefined (net ≤ 0); Technology 121% and top-10 155% of a near-zero gross total | **FAIL** (both defined flags fire; shares unstable on this denominator) |
| 7 | leg decomposition reported | long +2.42 / short −2.17 gross; not short-only-carried | reported |

Criterion 5 comparison, explicitly: the actual per-side cost estimate is
half-spread 2 bp + commission 1 bp = 3.0 bp/side (borrow is charged
separately and is excluded from the per-side break-even by construction);
the gate requires break-even ≥ 6.0 bp/side; measured break-even is
4.21 bp/side.

**Failed criteria 2, 4, and 5 → the termination rule applies: the live path
terminates. No re-tuning of windows, baselines, quantiles, or holding
periods.**

Known optimism in the cost model, all of which would make the true net
worse: short dividend cash flows are not modeled (adjusted-close prices
carry dividends only approximately), delisted names exit at their last
available close with no exit cost, and the universe is survivor-biased.
The failure verdict does not depend on any of these.

## Interpretation

The gate did exactly what it was frozen to do. C15's cross-sectional rank
effect is real at the research level (v0 sector-neutral incremental
t = −2.29, IC −0.011) but small — roughly 1.7%/yr gross on a decile
long-short — and it lives in a signal that decays within a month, which
forces ~86% monthly book turnover. At institutional-floor costs of
3 bp/side plus 50 bp borrow the drag is ~1.75%/yr, which consumes the gross
almost exactly; doubling costs makes the strategy clearly negative, and the
break-even margin (1.40x actual costs) leaves no room for real-world
slippage. Execution timing is immaterial (next_open ≈ close), factor
exposures are nil (R² 0.002), and the monthly net P&L series carries no
statistical evidence of positive drift (HAC t −0.01 against a ceiling of
2.57). The pre-registered conclusion stands: a real but untradeably small
predictive effect, reported as such — the research finding (C15 as the
strongest unconfirmed candidate below the multiple-testing ceiling) is
unchanged, and the live path is closed.
