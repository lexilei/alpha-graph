# alpha-graph

**Do SEC-filing text and inter-company graph structure carry cross-sectional
equity-return information incremental to price and volume?** A point-in-time
factor-research pipeline on the S&P 500 (499 names, 2011–2026), evaluated
monthly against 21-trading-day forward returns under a pre-registered protocol.

The contribution is the evaluation discipline as much as the factors: a
permanent factor registry, incremental-IC testing against a price/volume
control set, sector-neutral robustness checks, and a trial ledger that counts
every test for multiple-comparison accounting. Every number reproduces from
`data/cache/`.

## Method

Each factor receives a permanent ID (`FACTORS.md`) and is scored by the monthly
cross-sectional Spearman rank-IC between the signal and forward returns. The
decision statistic is the **incremental** IC — the signal residualized, each
month, against a six-factor price/volume control set (12-1 and short-horizon
momentum, realized volatility, and a dollar-volume size proxy) — so a factor is
credited only for information those cheap controls do not already carry. Each
result is reported raw and sector-neutral. Trials are logged in
`reports/factor_preregistration.md`; significance is read against the trial
count, not a naive t=2.

## Data

| Source | Coverage |
|---|---|
| 10-K | 7,646 filings |
| 10-Q | 21,487 filings |
| 8-K | 33,490 filings |
| Prices | 499 tickers, daily OHLCV, 2011–2026 |
| Company graph | 10,997 supplier/customer/competitor edges, LLM-extracted from 10-K business sections, each backed by a source sentence |

## Findings

### Text change — Lazy Prices (factor 1)

Year-over-year TF-IDF cosine between a firm's consecutive 10-Ks (Cohen, Malloy &
Nguyen, 2020): large language change signals subsequent underperformance.

| | monthly IC | t |
|---|---|---|
| standalone | +0.012 | 2.3 |
| incremental over price/volume controls | +0.007 | 1.5 |
| incremental, sector-neutral | +0.007 | 1.5 |

A real but modest effect. Roughly 40% of the standalone signal is subsumed by
price/volume controls; the incremental component is positive and stable across
sub-periods but below the pre-registered promotion threshold. This estimate is
the corrected one — an earlier build reported the factor as null (t≈0.7) until
an audit traced that to a filing-extraction contamination bug (fallback-mode
documents paired against structured ones, plus unfiltered amendments) that was
manufacturing spurious low-similarity pairs; fixing it moved the standalone t
from 0.7 to 2.3.

### Cross-firm momentum spillover (factor 20)

A firm's *customers'* prior-month return as a predictor of its *own* next-month
return (Cohen & Frazzini, 2008: value propagates slowly across economic links
because investor attention is siloed by firm). Computed over the directed
customer edges of the graph, high-confidence only.

| | monthly IC | t |
|---|---|---|
| standalone | +0.012 | 1.3 |
| incremental over price/volume controls | +0.014 | 2.0 |
| incremental, sector-neutral | +0.010 | 1.5 |

The strongest candidate in the registry. The incremental IC *rises* after
removing the firm's own momentum — consistent with a genuine lag effect rather
than a momentum proxy — and survives sector-neutralization, unlike a symmetric
all-edge-types variant that does not. It holds sign across both sample halves;
the quintile long/short is +0.32%/month (t=1.7), about one-fifth the magnitude
of the original study, consistent with a real but attenuated effect in
mega-caps two decades after the source sample.

**Status: unconfirmed.** The incremental t sits near the significance level
implied by the trial count, and the graph edges are LLM-extracted and therefore
not strictly point-in-time (a model reading a 2013 filing has forward
knowledge). Promotion is gated on an out-of-sample confirmation using
SEC-disclosed major-customer relationships (rule-based, point-in-time), which
is independent of the extraction method used to find the effect.

### Signals that did not survive

The symmetric graph-spillover variants (factors 18–19) and the embedding- and
tone-based text factors are tracked in `FACTORS.md` with their verdicts; the
embedding factors (11, 13, 14) are being re-evaluated on the corrected filing
inputs.

## Scope and caveats

- **Survivorship.** The universe is current index constituents; ~38% of names
  that passed through the S&P 500 over 2011–2026 are absent. This attenuates
  every IC reported here.
- **Graph edges** are LLM-extracted (mild forward-knowledge risk) and both
  endpoints are restricted to the S&P 500 — the regime in which the
  customer-momentum effect is weakest.
- **Size proxy.** A dollar-volume liquidity measure stands in for market cap
  pending point-in-time shares outstanding; sector labels are a current GICS
  snapshot.
- Results are reported as evaluated — one borderline text factor, one
  incremental-but-unconfirmed graph factor — without a deployable-strategy
  claim.

## Prior-results correction

Builds of this repository before 2026-06 reported a strategy-level result (a
long-only Sharpe near 0.8 and an FF5+momentum alpha at t≈3). An internal audit
traced that alpha to a one-month benchmark misalignment in the attribution
code; corrected, it is approximately zero. Those figures are withdrawn and are
not claimed here. The current results are the post-audit ones and reproduce
from cache.

## Reproduce

```bash
pip install -e ".[dev]"
cp .env.example .env          # SEC EDGAR identity; LLM key for graph extraction

# data
python scripts/download_filings_v2.py --forms 10-K 10-Q --start-year 2011 --end-year 2026
python -m alpha_graph.data.market --max-tickers 500 --years-back 15

# factors (full registry in FACTORS.md)
python -m alpha_graph.signals.lazy_prices                       # factor 1
python -m alpha_graph.data.relationships                        # LLM company graph
python -m alpha_graph.signals.graph_signal --customer-momentum  # factor 20

# evaluate
python scripts/factor_orthogonality.py evaluate --candidate cosine_similarity --accepted BASELINE
```

## Tests

```bash
pytest tests/ -q
```
