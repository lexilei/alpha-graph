# alpha-graph

Text factors from SEC filings (10-K, 10-Q, 8-K), evaluated cross-sectionally
on an S&P 500 panel (499 tickers, 2011–2026, monthly).

Results claimed by pre-2026-06 versions of this repo were retracted after an
internal audit and are not claimed here. Current state is below; every number
reproduces from `data/cache/`.

## Data

| Corpus | Files | Coverage |
|---|---|---|
| 10-K | 7,646 | 83–100% of tickers per year, 2012+ |
| 10-Q | 21,487 | 83–100% per year, 2011+ |
| 8-K | 33,490 | recency-skewed, backfill pending |
| Prices | 499 tickers daily OHLCV, 2011-04 → 2026-04 | complete |

Remaining per-year gaps are late IPOs, not missing downloads. Downloads are
manifest-checkpointed and resumable (`scripts/download_filings_v2.py`).

## Factors

Registered with permanent IDs in `FACTORS.md`. Selection protocol (IS/OOS
split, incremental-IC thresholds, trial counting for multiple-testing) is
pre-registered in `reports/factor_preregistration.md` — committed before
results were examined.

Evaluation tool: `scripts/factor_orthogonality.py` — residualizes a candidate
against the accepted set per monthly cross-section and tests whether the
residual still predicts 21-day forward returns.

## Current results

On the complete corpus, factor 1 (10-K TF-IDF cosine, the core Lazy Prices
measure) has a standalone monthly IC of +0.0037 (t = 0.68) — not significant.
The decile relation is U-shaped, so the paper's short side does not replicate
in this universe. Evaluation of the remaining text factors (tone shift,
embedding similarity, change detection, 10-Q YoY) is in progress under the
pre-registered protocol.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # SEC EDGAR identity

# data
python scripts/download_filings_v2.py --forms 10-K 10-Q --start-year 2011 --end-year 2026
python -m alpha_graph.data.market --max-tickers 500 --years-back 15

# factors
python -m alpha_graph.signals.lazy_prices                 # 1
python -m alpha_graph.signals.lazy_prices_10q             # 10
python -m alpha_graph.signals.embed_sim_10k --tag fin --model FinLang/finance-embeddings-investopedia  # 11
python -m alpha_graph.signals.tone_10k                    # 12
python -m alpha_graph.signals.embed_sim_10k --tag bge --model BAAI/bge-base-en-v1.5                    # 13
python -m alpha_graph.signals.change_detect_10k           # 14

# evaluate
python scripts/factor_orthogonality.py greedy
```

## Tests

```bash
pytest tests/ -q
```
