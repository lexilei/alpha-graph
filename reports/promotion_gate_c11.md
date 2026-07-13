# Promotion gate: C11-family daily strategy (pre-frozen 2026-07-13)

Frozen BEFORE any tradeable-backtest number exists. These criteria do not
move after results are seen; a failed gate terminates the live path — it does
not trigger parameter search.

## Scope

Candidate strategy: transparent, deterministic decile long-short from the
daily abnormal 8-K frequency signal (C15, `evt8k_freq_z_d`; short high
abnormal frequency), S&P 500 PIT universe, v0 conventions (PIT on, t+1 on,
daily grid). No ML combiner. C10 is excluded from the live path until it is
replicated on SEC-disclosed customer relationships with edge termination
dates; the LLM graph is not strictly PIT.

Reference numbers at freeze time (so attenuation is measurable): monthly C11
full-coverage sector-neutral incremental t = −2.81; monthly quintile L/S
+0.11%/mo (t≈1.2) full sample. The daily variant is untested — its first
numbers come from the pre-committed 2³ factorial. If the tradeable backtest
is built, it runs on the factorial's v0 cell, not on a tuned variant.

## Gate criteria (all must hold)

1. **Forward/paper period**: net P&L positive on data genuinely unused at
   design time (post-2026-04 paper period as it accrues, or a held-out slice
   never opened during development).
2. **Net Sharpe ≥ 0.8** after full costs: half-spread, commissions,
   regulatory fees, borrow, short dividends, at next-tradable-window prices
   (never same-close).
3. **Statistics survive adjustment**: HAC t on the net monthly P&L series
   clears the full-ledger multiple-testing ceiling (ledger N at evaluation
   time, `emax_null(2N)`); effective-N appears only as a redundancy
   descriptor, never as the denominator.
4. **Cost-doubling robustness**: still non-negative with all cost components
   AND borrow doubled.
5. **Break-even margin**: implied break-even cost ≥ 2× the actual cost
   estimate.
6. **No concentration**: profits not dominated by a single year, a single
   sector, or a handful of names — report the leave-one-year-out minimum,
   sector attribution, and top-10-name share; any of {one year > 60% of
   P&L, one sector > 50%, ten names > 50%} fails the gate.
7. **Leg decomposition reported**: long-leg and short-leg contributions
   separately. If only the short leg carries the effect, the strategy is
   graded short-only (borrow constraints then bind) — it is not presented
   as long-only alpha.

## Termination rule

If the v0-cell daily strategy fails gates 2, 4, or 5 on the first honest
pass, the live path terminates. No re-tuning of windows, baselines,
quantiles, or holding periods to rescue it — the full-sample monthly L/S was
~11 bp/mo; if daily attachment does not lift it past doubled costs, the
finding stays what it is: a real but untradeably small predictive effect,
reported as such in the README.

## Project-goal decision (recorded)

The project's primary deliverable is the empirical research answer (do
SEC-derived signals carry incremental cross-sectional information under
honest conventions?). The live path is a conditional extension gated by this
document, not the default. Gates 4–5 of the live plan (OMS, shadow/paper/
micro-live ladder) are reached only if every criterion above passes.
