# alpha-graph

Does SEC-filing text — or the graph of inter-company links buried in it —
carry stock-selection information *beyond what price and volume already price
in*? Tested on an S&P 500 panel (499 tickers, 2011–2026, monthly
cross-sections), target = next 21-trading-day return.

The answer so far is "barely, if at all." This repo is the honest measurement,
not a strategy.

## How factors are judged

Every factor gets a permanent ID (`FACTORS.md`) and is scored by monthly
cross-sectional Spearman rank-IC against forward returns — but the number that
decides anything is the **incremental** IC over a price/volume control set
(12-1 and short-horizon momentum, realized volatility, a dollar-volume size
proxy), residualized per month, with a sector-neutral robustness pass. A signal
that only "works" until you subtract cheap controls is a control in disguise.
Every trial is logged (`reports/factor_preregistration.md`) so multiple testing
is accounted for, and every number reproduces from `data/cache/` via
`scripts/factor_orthogonality.py`.

## Data

| Corpus | Volume |
|---|---|
| 10-K | 7,646 filings |
| 10-Q | 21,487 filings |
| 8-K | 33,490 filings |
| Prices | 499 tickers, daily OHLCV 2011–2026 |
| Company graph | 10,997 customer/supplier/competitor edges, LLM-extracted from 10-K business sections, each backed by a quoted sentence |

## Results

### Factor 1 — Lazy Prices (10-K text change)

Year-over-year TF-IDF cosine between a firm's consecutive 10-Ks
(Cohen-Malloy-Nguyen 2020): a small language change should mean no news, a large
one should mean trouble. On this universe:

| | monthly IC | t |
|---|---|---|
| standalone | +0.012 | 2.3 |
| incremental over price/volume controls | +0.007 | 1.5 |
| " , sector-neutral | +0.007 | 1.5 |

Real but weak. The standalone signal looks significant, but ~40% of it is
spanned by cheap price/volume controls, and the incremental effect does not
clear a multiple-testing-adjusted bar. The paper's short side (big-change firms
underperforming) does not replicate cleanly in large caps.

### Factor 20 — customer-momentum spillover (graph)

A firm's *customers'* past-month return predicting its *own* next-month return
(Cohen-Frazzini 2008: information crosses economic links slowly because
attention is siloed by company). Computed over the customer edges of the LLM
graph, high-confidence only, direction-aware.

| | monthly IC | t |
|---|---|---|
| standalone | +0.012 | 1.3 |
| incremental over price/volume controls | +0.014 | 2.0 |
| " , sector-neutral | +0.010 | 1.5 |

The registry's strongest surviving candidate: incremental IC of +0.014 that
gets *stronger* after removing the firm's own momentum, survives
sector-neutralization (unlike a symmetric all-edges version), and holds sign in
both sample halves. Quintile long/short is +0.32%/mo (t=1.7), ~1/5 of the
paper's original magnitude — plausible for mega-caps two decades post-sample.

**Reported as unconfirmed.** The t-stat sits near the noise ceiling implied by
the trial count, and the graph edges are LLM-extracted (a model reading a 2013
filing already knows what became important — mild hindsight). A clean
confirmation on *SEC-disclosed* major-customer links (rule-based, point-in-time)
is the planned next step; only then would it move from candidate to accepted.

## Limitations

- **Survivorship.** Universe is current constituents; ~38% of tickers that
  passed through the index over 2011–2026 are absent (departed members). This
  attenuates every IC here — the true effects are likely weaker still, or the
  measurements noisier.
- **Graph edges are LLM-extracted** (possible hindsight; not strictly
  point-in-time) and both endpoints are restricted to the S&P 500 — the weakest
  corner of the customer-momentum effect, which is strongest for small suppliers
  of large customers.
- **No true market cap** (a dollar-volume liquidity proxy stands in until
  point-in-time shares outstanding are wired in); sector labels are a current
  GICS snapshot, mildly non-point-in-time.
- **Nothing here clears a strict significance bar.** These are honest weak /
  borderline results, deliberately reported without a deployable-strategy claim.

## Retraction note

Pre-2026-06 versions of this repo claimed a strategy-level result (a long-only
Sharpe near 0.8 and a factor-model alpha with t≈3). An internal audit traced
that alpha to a one-month benchmark misalignment in the attribution code;
corrected, it is approximately zero. Those numbers are **retracted** and are not
claimed anywhere here. Everything above reproduces from cache.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env          # SEC EDGAR identity; LLM key for graph extraction

# data
python scripts/download_filings_v2.py --forms 10-K 10-Q --start-year 2011 --end-year 2026
python -m alpha_graph.data.market --max-tickers 500 --years-back 15

# factors (see FACTORS.md for the full registry)
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
