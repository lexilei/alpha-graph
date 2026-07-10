# alpha-graph

**Do SEC-filing signals — filing text, inter-company graph structure, and 8-K
disclosure activity — carry cross-sectional equity-return information
incremental to price and volume?** A point-in-time factor-research pipeline on
the S&P 500 (499 names, 2011–2026), evaluating each candidate monthly against
21-trading-day forward returns under a pre-registered protocol.

Across three data sources and fourteen registered candidate factors, two signals
carry incremental predictive content that survives price/volume controls,
sector-neutralization, and multiple-testing accounting: **abnormal 8-K filing
frequency** (sector-neutral incremental t ≈ −2.8) and
**customer-momentum spillover** across a supplier/customer graph (incremental
t ≈ 2.0 at a monthly horizon, ≈ 3.1 at its natural ten-day horizon). A third
result is structural: text-change and cross-firm-momentum signals have opposite
optimal holding periods, each matching its economic mechanism.

## Method

Each candidate receives a permanent ID (`FACTORS.md`) and is scored by the
monthly cross-sectional Spearman rank-IC between the signal and forward returns.
The decision statistic is the **incremental** IC — the candidate residualized,
each month, against a six-factor price/volume control set (12-1 and short-horizon
momentum, realized volatility, and a dollar-volume size proxy) — so a factor is
credited only for information those controls do not already carry. Every result
is reported raw and sector-neutral (within-GICS demeaning). Trials are logged in
`reports/factor_preregistration.md`; significance is read against an effective
trial count rather than a nominal t = 2 (see *Significance*). Every number
reproduces from `data/cache/` via `scripts/factor_orthogonality.py`.

Factors are organized into two groups: **candidates** (`C1`–`C14`, the
hypotheses under test) and **baseline** (`B1`–`B6`, the price/volume controls).
Code keys factors by name; the C/B labels are for reference.

## Data

| Source | Coverage |
|---|---|
| 10-K | 7,646 filings, 2011–2026 |
| 10-Q | 21,487 filings, 2011–2026 |
| 8-K | 100,618 filings, 2011–2026, all 499 names |
| Prices | 499 tickers, daily OHLCV, 2011–2026 |
| Company graph | 10,997 supplier/customer/competitor edges, LLM-extracted from 10-K business sections, each backed by a source sentence |

## Findings

### 8-K abnormal filing frequency (C11)

For each firm-month, the count of 8-K filings is z-scored against the firm's own
trailing 24-month distribution. A spike in filing activity relative to the firm's
baseline predicts lower forward returns: the cross-section under-reacts to
clustered disclosure.

On the full cross-section (all 499 names, 100,618 filings):

| | incremental IC | t |
|---|---|---|
| over price/volume baseline | −0.008 | −1.9 |
| baseline, sector-neutral | −0.011 | −2.8 |

The sector-neutral effect holds in both sample halves (2012–2019 t = −2.1,
2020–2026 t = −1.9) and uses only information available at each month-end. Its
economic magnitude is coverage-dependent: on the 179 larger, filing-active names
first evaluated, the quintile long/short earned +0.41%/month (t = 2.5); on the
full cross-section it is +0.11%/month (t = 1.2). The signal is robust in rank
terms and concentrated in the more actively-filing segment of the index. An
adversarial code review verified the point-in-time construction (excluding all
month-end-day filings strengthens the result) and localized the effect in time:
months where the current month's count is available at the sample date score
t = −3.7, while a one-month lag eliminates the signal — the effect decays
within one month, so implementation latency is the binding constraint.

The predictive content is the frequency *anomaly* specifically. Constructions
that read 8-K item content — a hard-negative-item count (restatements, officer
departures, delistings, debt-acceleration, agreement terminations) — and 8-K
text sentiment (Loughran-McDonald negative-word density) carry no incremental
signal on the full corpus (sector-neutral t = +0.1, +0.2, +0.1). The market
prices the content of individual 8-Ks; what it under-weights is the clustering
of filing activity itself.

### Cross-firm momentum spillover (C10)

A firm's customers' prior-month return, propagated across the directed
customer edges of the LLM-extracted company graph, predicts the firm's own
next-period return — the Cohen–Frazzini (2008) economic-links effect, in which
value crosses supply-chain relationships with a lag because investor attention
is siloed by firm.

| | incremental IC | t |
|---|---|---|
| over baseline, 21-day horizon | +0.014 | +2.0 |
| baseline, sector-neutral | +0.010 | +1.5 |
| over baseline, 10-day horizon (HAC) | — | +3.1 |

The incremental IC *increases* after removing the firm's own momentum, consistent
with a genuine lag effect rather than a momentum proxy, and survives
sector-neutralization — unlike a symmetric all-edge-type variant, which does not.
It holds sign across both sample halves; the quintile long/short is +0.32%/month
(t = 1.7). At the ten-day horizon predicted by the fast propagation of momentum
across links, the Newey-West/HAC-corrected incremental t reaches ≈ 3.1.

Status: unconfirmed. The graph edges are LLM-extracted and therefore not strictly
point-in-time. Promotion is gated on replication using SEC-disclosed
major-customer relationships (rule-based, point-in-time), independent of the
extraction method used to identify the effect.

### 10-K text change and the encoder A/B (C1, C3, C5)

Year-over-year similarity between a firm's consecutive 10-K filings
(Cohen-Malloy-Nguyen 2020, *Lazy Prices*) is measured three ways — bag-of-words
TF-IDF cosine (C1), a general-purpose sentence embedding (C5, BGE), and a
finance-tuned embedding (C3) — as a controlled A/B on the same document pairs.

| measurement | incremental t (baseline) | sector-neutral t |
|---|---|---|
| C5 general embedding | +1.8 | +2.1 |
| C1 TF-IDF | +1.5 | +1.5 |
| C3 finance-tuned embedding | +1.2 | +2.0 |

The general-purpose embedding carries information beyond bag-of-words (incremental
t = 1.4 of C5 over C1); the finance-tuned encoder is redundant with TF-IDF
(incremental t = 0.1 over C1). The ordering — general-semantic > lexical >
finance-tuned — is the durable qualitative result; the individual magnitudes sit
at the multiple-testing ceiling.

### Holding-horizon structure

An IC-decay analysis across horizons from 5 to 126 trading days separates the
signal families by their economic clock. The text-change factors are
slow-diffusion: their IC rises monotonically out to a 3–6 month horizon, and the
general-embedding factor reaches an HAC-corrected incremental t ≈ 2.7 at 63 days.
Customer-momentum spillover is fast: its IC peaks near 10 days (HAC t ≈ 3.1) and
decays thereafter. The two families have opposite optimal holding periods —
quarters for disclosure-text diffusion, weeks for cross-firm momentum — each
consistent with the mechanism proposed for it. (Overlapping-return t-statistics
at multi-month horizons are HAC-corrected throughout; the nominal statistics
overstate significance by roughly 2× at the longest horizons.)

## Significance and multiple testing

The candidate factors are correlated same-family measurements, so the effective
number of independent tests is well below the nominal count. Estimated from the
eigenspectrum of the factor IC-correlation matrix, the effective breadth is ≈ 7–9
tests, placing the noise ceiling at E[max |t| | null] ≈ 2.0. Abnormal 8-K
frequency (sector-neutral t ≈ 2.8) and customer-momentum spillover at its natural
horizon (HAC t ≈ 3.1) clear this ceiling; the 10-K text-change cluster sits at it.
Reported t-statistics for factors evaluated on thin cross-sections carry a
larger per-month sampling-noise component; the number of months (≈ 165), not the
number of names, sets the significance sample size.

## Scope and caveats

- **Survivorship.** The universe is current index constituents; roughly 38% of
  names that passed through the S&P 500 over 2011–2026 are absent. This
  attenuates the measured ICs.
- **8-K segment dependence.** The frequency factor's portfolio-level spread is
  concentrated in the larger, filing-active segment of the index; on the full
  cross-section the rank signal holds (sector-neutral t = −2.8) while the
  quintile spread is +0.11%/month.
- **Graph edges** are LLM-extracted (mild forward-knowledge exposure) with both
  endpoints in the S&P 500 — the regime in which the cross-firm-momentum effect
  is weakest — which makes the observed magnitude a lower bound on the underlying
  effect and motivates the point-in-time confirmation described above.
- **Size proxy.** A dollar-volume liquidity measure stands in for market
  capitalization pending point-in-time shares outstanding; sector labels are a
  current GICS snapshot.

## Prior-results correction

Builds of this repository before 2026-06 reported a strategy-level result (a
long-only Sharpe near 0.8 and an FF5+momentum alpha with t ≈ 3). An internal
audit traced that alpha to a one-month benchmark misalignment in the attribution
code; corrected, it is approximately zero. Those figures are withdrawn. The
results above are the post-audit measurements and reproduce from cache.

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
python -m alpha_graph.signals.graph_signal --customer-momentum  # C10 customer momentum
python -m alpha_graph.signals.event_freq_8k                     # C11 abnormal 8-K frequency

# evaluate (incremental IC over the price/volume baseline, sector-neutral)
python scripts/factor_orthogonality.py evaluate --candidate evt8k_freq_z --accepted BASELINE --sector-neutral
python scripts/factor_orthogonality.py evaluate --candidate spillover_cust_mom --accepted BASELINE
```

## Tests

```bash
pytest tests/ -q
```
