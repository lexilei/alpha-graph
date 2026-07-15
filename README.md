# alpha-graph

**Do SEC-filing signals — filing text, inter-company graph structure, and 8-K
disclosure activity — carry cross-sectional equity-return information
incremental to price and volume?** A factor-research pipeline on the S&P 500
(499 names, 2011–2026), evaluating each candidate monthly against
21-trading-day forward returns under a pre-registered protocol.

Current answer: not demonstrably, at this sample and trial count. A
pre-committed 2³ convention factorial (point-in-time membership ×
availability lag × signal grid; `reports/factorial_v0.md`) showed that for
the borderline factors the evaluation conventions were doing the work. Under
the frozen honest convention set (v0), the 10-K text-change factor and the
customer-momentum spillover factor lose their signal (sector-neutral
incremental t = +0.24 and +0.21); the abnormal-8-K-frequency family survives
attenuated (daily variant t = −2.29, monthly t = −2.00) but sits below the
full-ledger multiple-testing ceiling. Later the same day, earnings drift
(C17 SUE/PEAD, incremental t = +1.70 after an availability correction,
drift concentrated at ~1 month) and
clustered insider buying (C16, calendar-time FF5+MOM alpha +7.67%/yr,
HAC t +3.33) were evaluated under pre-pinned bars: C17 is a below-ceiling
candidate; C16 passed its pinned bar but a shifted-event placebo attributes
the alpha to portfolio composition rather than event timing, so its
ceiling-clearing claim was retracted the same day (see the ledger's
re-grade). The registry holds **0 accepted factors**, and no result is
currently treated as clearing the statistical bar (N = 133 tracked looks,
two-sided E[max] ceiling ≈ 2.86 — a mean, not a significance bar; the 5%
family-wise threshold is ≈ 3.55). A concurrency-conditioned insider variant
(C20) was registered, evaluated against a pre-pinned three-condition bar,
and rejected the same day.

## Method

Each of the twenty registered candidates receives a permanent ID
(`FACTORS.md`) and is scored by the monthly cross-sectional Spearman rank-IC
between the signal and forward returns. The decision statistic is the
**incremental** IC — the candidate residualized, each month, against a
six-factor price/volume control set (12-1 and short-horizon momentum,
realized volatility, and a dollar-volume size proxy) — so a factor is
credited only for information those controls do not already carry. Headline
numbers are sector-neutral (within-GICS demeaning of all rank series).

**Evaluation conventions are frozen (v0, 2026-07-13):** point-in-time S&P 500
membership, a one-calendar-day availability lag between a signal's
computation date and its first use, and daily-updated signal caches
(`reports/factorial_v0.md`). These are the code defaults; the judge's
`--no-pit` / `--no-daily` / `--lag 0` flags exist only for attribution work.
Every trial is logged in `reports/factor_preregistration.md`, and
significance for any external claim is read against the full ledger count via
a two-sided expected-max ceiling (see *Significance*). Every number
reproduces from `data/cache/` via `scripts/factor_orthogonality.py`.

Factors are organized into two groups: **candidates** (`C1`–`C20`, the
hypotheses under test) and **baseline** (`B1`–`B9` — the judge's default
set stays B1–B6; B7–B9 PIT fundamentals are opt-in via `--accepted
EXTENDED`). Code keys factors by name; the C/B labels are for reference.

## Data

| Source | Coverage |
|---|---|
| 10-K | 7,646 filings, 2011–2026 |
| 10-Q | 21,487 filings, 2011–2026 |
| 8-K | 100,618 filings, 2011–2026, all 499 names |
| Prices | 503 tickers, daily OHLCV, 2011–2026 (spinoff-seam registry: `data/cache/market_data_seams.parquet`) |
| Company graph | 11,025 extracted relationship rows (3,113 unique directed edges after per-pair dedup), supplier/customer/competitor, LLM-extracted from 10-K business sections, each backed by a source sentence |

### Paid-data staging

No commercial dataset has been purchased. The approved procurement path is
documented in `reports/delisted_data_decision.md`. Its unauthenticated download
plan can be inspected without a subscription:

```bash
python -m alpha_graph.data.sharadar plan --profile validation --start 2009-01-01
```

After written license confirmation, `fetch` requires both
`NASDAQ_DATA_LINK_API_KEY` and an explicit `--license-expires` date. Raw and
schema-validated Parquet snapshots remain isolated under `data/raw/sharadar/`
and `data/cache/sharadar/`; they never overwrite `market_data.parquet`. Blind
identity and coverage QA runs with:

```bash
python -m alpha_graph.data.sharadar_qa --snapshot SNAPSHOT_ID
```

The QA command exits nonzero until all machine gates and high-risk manual
identity adjudications pass.

## Findings

### The conventions were doing the work

A pre-committed 2³ factorial varied point-in-time membership (on/off),
availability lag (0/1 calendar day), and signal grid (month-end vs daily
caches) for the four flagship factors — 19 informative cells, each a
ledgered look (`reports/factorial_v0.md`). Sector-neutral incremental t over
the six-factor baseline, 165–168 months:

| factor | pre-PIT | v0 | attribution |
|---|---|---|---|
| C1 10-K text change (TF-IDF) | +1.48 | **+0.24** | PIT main effect −1.21; lag and grid verified inert |
| C10 customer-momentum spillover | +1.40 | **+0.21** | PIT −0.69 and t+1 lag −0.55 compound |
| C11 8-K abnormal frequency, monthly | −2.81 | **−2.00** | PIT only (lag and grid verified inert for it) |
| C15 8-K abnormal frequency, daily | −3.29 / −3.67 | **−2.29** | PIT attenuates; the t+1 lag *strengthens* it |

The mechanism behind the PIT effect is identified: point-in-time filtering
*raises* every factor's candidate coverage share (index non-members are
sparse in filings and graph edges), and the pre-PIT borderline cluster — C1
at +1.5, C5 at +2.1, C10 at +1.5–2.0 — was carried by non-member rows. A
sensitivity cell holding the price/volume controls to the same t+1 standard
moves C15 by −0.04 (robust at −2.33) and moves C1/C10 to +0.10/−0.05.

Re-evaluated on the same v0 panel (two further ledgered looks), the
remaining open text candidates fall with C1: C5 general-embedding similarity
+2.08 → +1.43, C6 new-content fraction −1.72 → −0.73. C1, C5, C6, and C10
are rejected under v0. C10's rejection keeps its pre-registered path back:
replication on SEC-disclosed major-customer relationships (rule-based,
point-in-time), independent of the LLM extraction that found the effect.
Pre-PIT diagnostics — the encoder A/B ordering (general-semantic > lexical >
finance-tuned) and the opposite IC-decay horizons of the text and graph
families (text HAC t ≈ 2.7 at 63 days, spillover ≈ 3.1 at 10 days) — are
attribution-era observations on the inflated panel; none is re-established
under v0.

### 8-K abnormal filing frequency (C11 monthly, C15 daily)

For each firm, the count of 8-K filings over a trailing window is z-scored
against the firm's own history; a spike in filing activity predicts lower
forward returns. Two registered variants: C11 (calendar-month count vs a
trailing 24-month baseline, month-end grid) and C15 (trailing
21-trading-day count vs a lagged non-overlapping baseline, daily grid —
registered before evaluation, scored as its own trial).

v0 numbers (sector-neutral incremental IC over the baseline, 168 months,
~386 names per month):

| | incremental IC | t |
|---|---|---|
| C15 daily | −0.0113 | −2.34 |
| C11 monthly | −0.0090 | −2.02 |

(Post-price-repair quotes, 2026-07-14; the factorial-era values −2.29/−2.00
remain in the table above as the v0-freeze record.)

C15 is consistent with the mechanism an adversarial review isolated in C11:
the signal lives in the current, still-fresh window (pre-PIT, fresh-attach
months scored t = −3.7 while a one-month lag erased it). Under the one-day
availability lag C15 *strengthens* (−1.85 → −2.29 under PIT) — same-close
attachment diluted the signal rather than carrying it. It is insensitive to
control timing (−2.33 with controls lagged) and has 95% coverage. The
predictive content is the frequency anomaly specifically: constructions
reading 8-K item content (hard-negative items) or tone (Loughran-McDonald
sentiment) carried nothing even pre-PIT (sector-neutral |t| ≤ 1.0).

Both variants sit below the full-ledger ceiling (≈ 2.86 at the current
count). Status:
candidates, unconfirmed; no significance claim is made.

### C15 tradeable backtest (gate-2): failed, live path closed

A pre-registered decile long-short on C15 — long the lowest-z decile, short
the highest, rebalanced monthly at the month-end close on that day's
available signal (v0 panel), at 2 bp half-spread + 1 bp commission per side
and 50 bp p.a. borrow — was run once, 2011-12 → 2026-04, against the frozen
promotion gate (`reports/promotion_gate_c11.md`; full results in
`reports/gate2_c15_tradeable.md`). The gross spread is ≈ +1.74%/yr, but the
signal's within-month decay forces ~86% monthly book turnover (20.7x/yr
one-way), and base-case costs consume the gross almost exactly: net Sharpe
**−0.00**. With every cost component doubled the strategy is clearly
negative (−1.75%/yr, Sharpe −0.27), and the break-even cost of 4.21 bp/side
is 1.40x the actual 3 bp/side estimate, below the gate's required 2x margin.
Gate criteria 2, 4, and 5 fail, so the pre-committed termination rule
applies: the live path is closed, with no re-tuning of windows, baselines,
quantiles, or holding periods. The research finding is unchanged — a real
but untradeably small predictive effect.

### LightGBM combiner (scope)

The walk-forward combiner (`ml_combiner.py`) predicts 21-day returns from C1
plus four price/volume controls; no SEC candidate beyond C1 was ever in its
feature set. Its out-of-sample attribution — baseline-only Sharpe 1.02 ≥
full 0.98, i.e. adding C1 does not help — predates the v0 conventions. It
shows the combiner is a momentum machine; it is not a test of the SEC
signals.

## Significance and multiple testing

As of 2026-07-14, N = 133 tracked ledger looks record a computed evaluation
statistic (registrations and infrastructure notes excluded; superseded rows
count — they were looks, as do the 12 back-filled looks from the 2026-07-10
IC-decay sweep). Selection is over |t|, so the reference bar for any
external claim is the two-sided expected-max ceiling E[max |t| | null] =
`emax_null(2N)` = **2.86** — with two review caveats now recorded in the
ledger: the E[max] ceiling is a mean, not a significance bar (the 5%
family-wise threshold at this N is ≈ 3.4), and row-based counting
understates statistic-level looks (≈ 150–185, ceiling ≈ 2.9). No result is
currently treated as clearing the bar: C16's +3.33 was re-graded after a
composition placebo (see the ledger), and C17 +1.70 / C15 −2.34 / C11
−2.00 sit below it. The candidates are correlated same-family
measurements: an eigenspectrum estimate puts the effective number of
independent bets at ≈ 7–9. That effective-N describes redundancy — how many
distinct ideas were really tried — and is never the significance
denominator. The number of months (165–168), not the number of names, sets
each statistic's sample size.

## Scope and caveats

- **Survivorship.** v0 applies point-in-time index membership, which removes
  forward bias (names enter the panel only while index members) but not
  survivor bias: roughly 38% of names that passed through the index over
  2011–2026 are absent entirely (departed-name gap 194 in 2011 → 23 in
  2025). The resulting ICs are survivor-conditioned; the direction of the
  bias is not known in advance.
- **8-K economics.** The quintile-spread magnitudes on record (+0.41%/month
  on the 179-name subset, +0.11%/month at full coverage, for C11) are
  pre-PIT. v0 portfolio-level economics were measured by the gate-2
  tradeable backtest (2026-07-13, see the C15 section): the daily grid
  recovers ≈ +1.74%/yr gross on a decile L/S, but the same within-month
  decay that motivated it makes turnover cost the binding constraint — net
  ≈ zero at base costs. A post-run audit found that stale scores could cross
  PIT membership exits. The figures remain the registered historical record
  but are not an exact PIT estimate; the failed live path remains closed.
- **Graph edges** are LLM-extracted (mild forward-knowledge exposure) with
  both endpoints in the S&P 500 — the regime where the cross-firm momentum
  effect is documented to be weakest. The SEC-disclosed-customer replication
  is C10's pre-registered path back.
- **Size proxy.** A dollar-volume liquidity measure stands in for market
  capitalization pending point-in-time shares outstanding; sector labels are
  a current GICS snapshot.

## Prior-results correction

Builds of this repository before 2026-06 reported a strategy-level result (a
long-only Sharpe near 0.8 and an FF5+momentum alpha with t ≈ 3). An internal
audit traced that alpha to a one-month benchmark misalignment in the
attribution code; corrected, it is approximately zero. Those figures are
withdrawn. Separately, factor statistics quoted before 2026-07-13 were
computed under pre-v0 conventions (current-constituent panel, same-close
attachment); they are superseded by the v0 numbers above and appear here
only as labeled attribution context.

## Reproduce

```bash
pip install -e ".[dev]"
cp .env.example .env          # SEC EDGAR identity; LLM key for graph extraction

# data
python scripts/download_filings_v2.py --forms 10-K 10-Q 8-K --start-year 2011 --end-year 2026
python -m alpha_graph.data.market --max-tickers 500 --years-back 15

# factors (full registry in FACTORS.md)
python -m alpha_graph.signals.lazy_prices                       # C1  10-K TF-IDF
python -m alpha_graph.signals.embed_sim_10k --tag bge --model BAAI/bge-base-en-v1.5   # C5
python -m alpha_graph.data.relationships                        # LLM company graph
python -m alpha_graph.signals.graph_signal --customer-momentum --daily  # C10 daily cache
python -m alpha_graph.signals.event_freq_8k                     # C11 monthly 8-K frequency
python -m alpha_graph.signals.event_freq_8k --daily             # C15 daily 8-K frequency

# evaluate (defaults are the frozen v0 conventions: PIT, lag 1, daily grid;
# --no-pit / --no-daily / --lag 0 are for attribution work only)
python scripts/factor_orthogonality.py evaluate --candidate evt8k_freq_z_d --accepted BASELINE --sector-neutral  # C15
python scripts/factor_orthogonality.py evaluate --candidate evt8k_freq_z --accepted BASELINE --sector-neutral    # C11
```

## Monitor

A read-only dashboard over the repo's own artifacts — the registry, the
ledger's accounting, and the parquet cache:

```bash
python -m alpha_graph.monitor            # build reports/dashboard.html + open
python -m alpha_graph.monitor --serve    # localhost:8765, rebuilt on each refresh
```

It shows the standings against the selection bars, the accounting history (N
and the ceiling at each refresh), the candidate registry filtered by status,
and the cache inventory (row counts, date spans, write times) read from parquet
footers. `--serve` re-reads everything per request, so it can stay open while a
build runs.

The headline accounting is read from the LATEST `Accounting refresh` row of
`reports/factor_preregistration.md` — the ledger's own rule for where the
authoritative count lives. The monitor deliberately does not recount looks: N
excludes registrations, infrastructure notes, and `role=null` controls by hand,
so a naive row count disagrees with it by construction. It computes nothing
except one cross-check — that each refresh row's stated ceiling still
reproduces from `ic_tools.emax_null` on its stated trial count, which
`tests/test_monitor.py` also asserts over every historical row. Adjudications
travel with their statistic: C16's +3.33 exceeds the 2.89 ceiling and is
labelled, everywhere it appears, with the ledger's "NOT treated as clearing —
composition".

## Tests

```bash
pytest tests/ -q
```
