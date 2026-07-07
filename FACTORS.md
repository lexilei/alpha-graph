# Factor Registry

Canonical, numbered list of factors. **IDs are permanent**: once assigned, a
number always means the same factor. New factors append the next free ID;
numbers are never reused or renumbered, even if a factor is rejected. Refer to
factors by ID everywhere else (logs, reports, commit messages).

Next free ID: **15**

Status legend: `candidate` (under evaluation) · `accepted` (in the model) ·
`rejected` (tested, no incremental signal — ID retained as a tombstone) ·
`inactive` (not a stock-picking candidate — e.g. market-level, no
cross-sectional content; excluded from the candidate pool, may still serve as
a conditioning/interaction input).

Prediction target for all factors: `fwd_return_21d` (next 21 trading-day return).

| ID | name | source | status | definition |
|----|------|--------|--------|------------|
| 1 | `cosine_similarity` | 10-K | candidate | TF-IDF (1–2gram, 10k vocab, EN stopwords) cosine between a firm's consecutive **same-type** 10-K filings; vectorizer refit per pair. High = little language change. |
| 2 | `event_score` | 8-K | candidate | Exponentially-weighted avg (λ=0.9/month, 12-month lookback) of per-filing 8-K item prior scores; each filing scored by its most extreme item (`max\|score\|`). |
| 3 | `event_count` | 8-K | candidate | Number of 8-K filings in the lookback window. |
| 4 | `regime_state` | HMM | inactive | 3-state Gaussian HMM label on market-level features (0 trending / 1 mean-reverting / 2 crisis). **Market-level — identical across the cross-section on a given date → no stock-picking content.** Conditioning/interaction use only. |
| 5 | `regime_prob` | HMM | inactive | Posterior probability of the current HMM state. **Market-level** (see ID 4). |
| 6 | `momentum_21d` | price | candidate | `close.pct_change(21)`. |
| 7 | `momentum_5d` | price | candidate | `close.pct_change(5)`. |
| 8 | `volatility_21d` | price | candidate | 21-day rolling std of daily returns × √252 (annualized realized vol). |
| 9 | `volume_zscore` | volume | candidate | `(volume − 63d mean) / 63d std`. |
| 10 | `cos_10q_yoy` | 10-Q | candidate | Lazy Prices on 10-Q, **year-over-year** (paper-faithful): TF-IDF cosine vs the 10-Q ~365 days earlier (same fiscal quarter, controls seasonality), same vectorizer as factor 1. `signals/lazy_prices_10q.py`. |
| 11 | `embed_sim_10k` | 10-K | candidate | **Semantic** similarity between consecutive 10-Ks: cosine of sentence-transformer embeddings, finance-tuned model `FinLang/finance-embeddings-investopedia` (pinned, local, deterministic; long sections chunked + mean-pooled). The modern alternative to factor 1's bag-of-words TF-IDF — added to A/B test whether finance-semantic embeddings beat TF-IDF (incremental IC of 11 over 1). |
| 12 | `tone_shift_10k` | 10-K | candidate | **Tone/direction** axis: change in Loughran-McDonald financial-sentiment word proportions (negative, uncertainty) vs the prior 10-K. Captures direction of change, which cosine ignores. PIT-safe (lexicon lookup, deterministic). |
| 13 | `embed_sim_10k_bge` | 10-K | candidate | Same as factor 11 but with the **general-purpose** `BAAI/bge-base-en-v1.5` encoder. Forms a 3-way A/B with factors 1 (TF-IDF) and 11 (finance-tuned): bag-of-words vs general-semantic vs finance-semantic similarity for the same 10-K change. |
| 14 | `new_content_frac` | 10-K | candidate | **Change detection**, not similarity: align the new 10-K's paragraph chunks against the prior filing's, count the share with no good match (cosine < thresh) → fraction of genuinely **added** content. Targets the saturation that defeats 11/13 (a single new paragraph survives instead of being pooled away). Embedding-space version of Lazy Prices "added text". `signals/change_detect_10k.py`. |

## Known issues (carry into any evaluation)

- **1** — corpus backfilled 2026-06-30 (7,146 pairs, 82–98% coverage 2012+).
  Standalone monthly IC on the full corpus: +0.0037, t = 0.68 (insignificant).
  Per-pair TF-IDF vocab refit means levels are not strictly comparable across
  pairs; cross-sectional ranks are what get used.
- **2, 3** — item prior scores are hand-coded, never calibrated; the cached
  `event_signals.parquet` is a dateless 102-row snapshot (unusable — recompute
  with `--timeseries` after an 8-K backfill).
- **4, 5** — market-level: zero cross-sectional ranking content. Conditioning/
  interaction use only.
- **10** — computed on the complete 10-Q corpus (19,913 YoY pairs, 498 tickers).
- **11, 13** — pretrained encoders (mild non-PIT); pin model versions. Purpose
  is the 3-way A/B vs factor 1, not a standalone claim. Whole-doc similarity
  saturates near 0.99 (motivated factor 14).
- **12** — PIT-safe (static lexicon).
- **14** — MATCH_THRESH and MAX_CHUNKS=150 (truncates very long filings) are
  hyperparameters; count variants toward N.
- Universe is current-constituent (survivorship): PIT membership filter not yet
  applied to factor evaluation.

## Convention

- A factor enters as `candidate`. Promote to `accepted` only after it clears the
  pre-registered incremental-IC threshold over the current accepted set
  (`scripts/factor_orthogonality.py`), confirmed out-of-sample.
- Rejected factors keep their ID and a one-line reason; do not delete the row.
- When adding a factor, also record its hypothesis and decision rule in the
  pre-registration log before running the test.
