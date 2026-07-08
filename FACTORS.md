# Factor Registry


Next free ID: **18**

Status legend: `candidate` (under evaluation) · `accepted` (in the model) ·
`rejected` (tested, no incremental signal — ID retained as a tombstone) ·
`baseline` (cheap price/volume control — text candidates must add incremental
IC over these).

Prediction target for all factors: `fwd_return_21d` (next 21 trading-day return).

| ID | name | source | status | definition |
|----|------|--------|--------|------------|
| 1 | `cosine_similarity` | 10-K | candidate | TF-IDF (1–2gram, 10k vocab, EN stopwords) cosine between a firm's consecutive **same-type** 10-K filings; vectorizer refit per pair. High = little language change. |
| 6 | `momentum_21d` | price | baseline | `close.pct_change(21)`. |
| 7 | `momentum_5d` | price | baseline | `close.pct_change(5)`. |
| 8 | `volatility_21d` | price | baseline | 21-day rolling std of daily returns × √252 (annualized realized vol). |
| 9 | `volume_zscore` | volume | baseline | `(volume − 63d mean) / 63d std`. |
| 10 | `cos_10q_yoy` | 10-Q | candidate | Lazy Prices on 10-Q, **year-over-year** (paper-faithful): TF-IDF cosine vs the 10-Q ~365 days earlier (same fiscal quarter, controls seasonality), same vectorizer as factor 1. `signals/lazy_prices_10q.py`. |
| 11 | `embed_sim_10k` | 10-K | candidate | **Semantic** similarity between consecutive 10-Ks: cosine of sentence-transformer embeddings, finance-tuned model `FinLang/finance-embeddings-investopedia` (pinned, local, deterministic; long sections chunked + mean-pooled). The modern alternative to factor 1's bag-of-words TF-IDF — added to A/B test whether finance-semantic embeddings beat TF-IDF (incremental IC of 11 over 1). |
| 12 | `tone_shift_10k` | 10-K | candidate | **Tone/direction** axis: change in Loughran-McDonald financial-sentiment word proportions (negative, uncertainty) vs the prior 10-K. Captures direction of change, which cosine ignores. PIT-safe (lexicon lookup, deterministic). |
| 13 | `embed_sim_10k_bge` | 10-K | candidate | Same as factor 11 but with the **general-purpose** `BAAI/bge-base-en-v1.5` encoder. Forms a 3-way A/B with factors 1 (TF-IDF) and 11 (finance-tuned): bag-of-words vs general-semantic vs finance-semantic similarity for the same 10-K change. |
| 14 | `new_content_frac` | 10-K | candidate | **Change detection**, not similarity: align the new 10-K's paragraph chunks against the prior filing's, count the share with no good match (cosine < thresh) → fraction of genuinely **added** content. Targets the saturation that defeats 11/13 (a single new paragraph survives instead of being pooled away). Embedding-space version of Lazy Prices "added text". `signals/change_detect_10k.py`. |
| 15 | `cos_latest_filing` | 10-K+10-Q | rejected | **Paper-faithful combined stream** (CMN 2020): at each date, the YoY same-type TF-IDF cosine of the firm's most recent periodic filing — union of factor 1's 10-K pairs and factor 10's 10-Q pairs, raw scores pooled (same vectorizer, comparable levels), as-of merged, no staleness cap. Pure construction from cached 1+10. **Rejected 2026-07-07**: IS 2012–2020 standalone IC t=0.83, incremental over price/volume baseline t=0.64 (rule needs t>3); OOS never touched. |
| 16 | `momentum_252_21` | price | baseline | Classic 12-1 momentum: `close(t-21)/close(t-252) − 1` (skip the most recent month). Added 2026-07-07 — the June audit found the cosine signal resembles exactly this, so it belongs in the controls. |
| 17 | `log_dollar_volume` | price+volume | baseline | Size/**liquidity proxy**: log of 63d median dollar volume. NOT true market cap (no PIT shares outstanding); its job is to catch text factors that secretly rank by company size. |

IDs 2–5 (8-K event factors, HMM regime factors) were retired untested and
removed from the codebase 2026-07-07; see git history. The IDs stay reserved.

Sector controls: `build_feature_panel` attaches a `sector` column (current GICS
snapshot, `sector_map.parquet` — mildly non-PIT). `factor_orthogonality.py
--sector-neutral` demeans all rank-z series within sector per month; it is a
control mode, not a factor.

## Known issues (carry into any evaluation)

- **1** — corpus backfilled 2026-06-30 (7,146 pairs, 82–98% coverage 2012+).
  Standalone monthly IC on the full corpus: +0.0037, t = 0.68 (insignificant).
  Per-pair TF-IDF vocab refit means levels are not strictly comparable across
  pairs; cross-sectional ranks are what get used.
- **10** — computed on the complete 10-Q corpus (19,913 YoY pairs, 498 tickers).
- **11, 13** — pretrained encoders (mild non-PIT); pin model versions. Purpose
  is the 3-way A/B vs factor 1, not a standalone claim. Whole-doc similarity
  saturates near 0.99 (motivated factor 14).
- **12** — PIT-safe (static lexicon).
- **14** — MATCH_THRESH and MAX_CHUNKS=150 (truncates very long filings) are
  hyperparameters; count variants toward N.
- **15** — pooling raw cosines across form types follows the paper; the
  no-staleness-cap choice is a (mild) variant — S&P 500 firms file ~quarterly,
  so scores are at most ~4 months old in practice. IS 2012–2020 result: decile
  L/S (long non-changers / short changers) +0.28%/mo, t=1.35 — sign matches the
  paper, magnitude in the paper's range, but indistinguishable from noise; the
  short side is absent (bottom decile ≈ average), spread comes from the long
  tail. Consistent with factor 1's U-shape finding.
- Universe is current-constituent (survivorship): PIT membership filter not yet
  applied to factor evaluation.

## Convention

- A factor enters as `candidate`. Promote to `accepted` only after it clears the
  pre-registered incremental-IC threshold over the current accepted set
  (`scripts/factor_orthogonality.py`), confirmed out-of-sample.
- Rejected factors keep their ID and a one-line reason; do not delete the row.
- When adding a factor, also record its hypothesis and decision rule in the
  pre-registration log before running the test.
