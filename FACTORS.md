# Factor Registry


Next free ID: **21**

Status legend: `candidate` (under evaluation) · `accepted` (in the model) ·
`rejected` (tested, no incremental signal — ID retained as a tombstone) ·
`baseline` (cheap price/volume control — text candidates must add incremental
IC over these).

Prediction target for all factors: `fwd_return_21d` (next 21 trading-day return).

| ID | name | source | status | definition |
|----|------|--------|--------|------------|
| 1 | `cosine_similarity` | 10-K | candidate | TF-IDF (1–2gram, 10k vocab, EN stopwords) cosine between a firm's consecutive **same-type** 10-K filings; vectorizer refit per pair. High = little language change. **Re-evaluated on clean inputs 2026-07-08** (after contamination fix, commit `0d930e0`): standalone IC +0.012, t=2.3; incremental over full price/volume baseline t=1.5, sector-neutral t=1.5 (168m). Borderline — below the promotion bar but no longer the pre-fix t=0.68 null; the strongest text factor after cleaning. Final call pending the family R1 re-run. |
| 6 | `momentum_21d` | price | baseline | `close.pct_change(21)`. |
| 7 | `momentum_5d` | price | baseline | `close.pct_change(5)`. |
| 8 | `volatility_21d` | price | baseline | 21-day rolling std of daily returns × √252 (annualized realized vol). |
| 9 | `volume_zscore` | volume | baseline | `(volume − 63d mean) / 63d std`. |
| 10 | `cos_10q_yoy` | 10-Q | rejected | Lazy Prices on 10-Q, **year-over-year** (paper-faithful): TF-IDF cosine vs the 10-Q ~365 days earlier (same fiscal quarter, controls seasonality), same vectorizer as factor 1. `signals/lazy_prices_10q.py`. **Rejected 2026-07-08 (R1), clean inputs**: standalone t=+1.35, vs full baseline t=+0.68, sector-neutral t=+0.25. |
| 11 | `embed_sim_10k` | 10-K | candidate | **Semantic** similarity between consecutive 10-Ks: cosine of sentence-transformer embeddings, finance-tuned model `FinLang/finance-embeddings-investopedia` (pinned, local, deterministic; long sections chunked + mean-pooled). The modern alternative to factor 1's bag-of-words TF-IDF — added to A/B test whether finance-semantic embeddings beat TF-IDF (incremental IC of 11 over 1). **Pre-fix numbers withdrawn (2026-07-08)**: the contamination fix invalidated the embedding cache; regeneration in progress, magnitude A/B to be re-run. |
| 12 | `tone_shift_10k` | 10-K | rejected | **Tone/direction** axis: change in Loughran-McDonald financial-sentiment word proportions (negative, uncertainty) vs the prior 10-K. Captures direction of change, which cosine ignores. PIT-safe (lexicon lookup, deterministic). **Rejected 2026-07-08 (R1), clean inputs**: standalone t=+1.06, vs baseline t=+0.72, sector-neutral t=+0.55 — spanned by price/volume controls. |
| 13 | `embed_sim_10k_bge` | 10-K | rejected | Same as factor 11 but with the **general-purpose** `BAAI/bge-base-en-v1.5` encoder. Forms a 3-way A/B with factors 1 (TF-IDF) and 11 (finance-tuned): bag-of-words vs general-semantic vs finance-semantic similarity for the same 10-K change. **Pre-fix numbers withdrawn (2026-07-08)**: embedding cache invalidated by the contamination fix; regeneration in progress. |
| 14 | `new_content_frac` | 10-K | rejected | **Change detection**, not similarity: align the new 10-K's paragraph chunks against the prior filing's, count the share with no good match (cosine < thresh) → fraction of genuinely **added** content. Targets the saturation that defeats 11/13 (a single new paragraph survives instead of being pooled away). Embedding-space version of Lazy Prices "added text". `signals/change_detect_10k.py`. **Pre-fix numbers withdrawn (2026-07-08)**: change-detection cache invalidated by the contamination fix; regeneration in progress. |
| 15 | `cos_latest_filing` | 10-K+10-Q | rejected | **Paper-faithful combined stream** (CMN 2020): at each date, the YoY same-type TF-IDF cosine of the firm's most recent periodic filing — union of factor 1's 10-K pairs and factor 10's 10-Q pairs, raw scores pooled (same vectorizer, comparable levels), as-of merged, no staleness cap. Pure construction from cached 1+10. **Rejected 2026-07-08, clean inputs** (re-derived from clean factors 1+10): standalone t=+1.41, vs full baseline t=+0.64, sector-neutral t=+0.01. |
| 16 | `momentum_252_21` | price | baseline | Classic 12-1 momentum: `close(t-21)/close(t-252) − 1` (skip the most recent month). Added 2026-07-07 — the June audit found the cosine signal resembles exactly this, so it belongs in the controls. |
| 17 | `log_dollar_volume` | price+volume | baseline | Size/**liquidity proxy**: log of 63d median dollar volume. NOT true market cap (no PIT shares outstanding); its job is to catch text factors that secretly rank by company size. |
| 18 | `spillover_event` | graph | rejected | **Cross-firm propagation** (Cohen–Frazzini 2008 style): confidence- and relation-weighted average of graph neighbors' 8-K event scores (supplier 1.0 / customer 0.8 / partner 0.5 / competitor −0.3; in-edge relations flipped to the target's perspective). Graph: 10,997 LLM-extracted edges (DeepSeek-V4-Pro) from 7,646 10-K Business sections, 2011–2026, both endpoints S&P 500. NaN = no scored neighbors. Month-end grid from `graph_spillover.parquet` (built on the `feat/graph-signal` branch). **Rejected 2026-07-08**: standalone t=−0.95 (wrong sign), vs full baseline t=−0.68, sector-neutral t=+0.12. |
| 19 | `spillover_momentum` | graph | rejected | Same propagation as 18 but of neighbors' 5-day momentum — the closest analogue to the paper's customer-momentum. Same graph, same weights, same caveats. **Rejected 2026-07-08**: standalone t=+0.48; incremental over full baseline t=+1.34 (sign matches C-F but noise-compatible for a low-prior family); sector-neutral halves it to t=+0.64 — part of the weak effect is sector momentum. |
| 20 | `spillover_cust_mom` | graph | candidate | **C-F-faithful asymmetric variant**: confidence-weighted mean of the firm's CUSTOMERS' 21-day momentum only (customers = out-`customer` edges + in-`supplier` counterparts, confidence ≥ 0.8 — the extraction rubric's "explicitly named" tier, set a priori). The paper's effect is customer→supplier with a ~1-month lag; 18/19's symmetric four-relation average dilutes it. NaN if no qualifying customers. Registered 2026-07-08 before computation. **First look 2026-07-08 — the registry's only live candidate**: incr over full baseline t=+1.97 (IC +0.0142), sector-neutral t=+1.54, both split-halves positive, quintile L/S +0.32%/mo (t=1.69). Not significant (thin xs ≈ 182; project ledger N≈10 puts E[max t | null] near this level) — needs an out-of-design confirmation before promotion. |

IDs 2–5 (8-K event factors, HMM regime factors) were retired untested and
removed from the codebase 2026-07-07; see git history. The IDs stay reserved.

Sector controls: `build_feature_panel` attaches a `sector` column (current GICS
snapshot, `sector_map.parquet` — mildly non-PIT). `factor_orthogonality.py
--sector-neutral` demeans all rank-z series within sector per month; it is a
control mode, not a factor.

## Known issues (carry into any evaluation)

- **1** — corpus backfilled 2026-06-30 (7,146 pairs, 82–98% coverage 2012+).
  Standalone monthly IC on the clean corpus (post contamination-fix, commit
  `0d930e0`): +0.012, t = 2.3. (The pre-fix +0.0037/t=0.68 was on contaminated
  inputs — fallback-mode extraction + unfiltered amendments; see git history.)
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
- **18, 19** — edges are LLM-extracted (the model knows post-filing history:
  mild non-PIT), both endpoints restricted to current S&P 500 (survivorship;
  the C-F effect is strongest outside mega-caps → prior LOW). Dual-class
  listings (GOOG/GOOGL, FOX/FOXA...) count as separate nodes. 18 inherits the
  8-K corpus's ticker sparsity (~180 active names per era). Business text
  truncated at 20k chars — long competition sections (e.g. INTC) fall past
  the cutoff.
- Universe is current-constituent (survivorship): PIT membership filter not yet
  applied to factor evaluation.

## Convention

- A factor enters as `candidate`. Promote to `accepted` only after it clears the
  pre-registered incremental-IC threshold over the current accepted set
  (`scripts/factor_orthogonality.py`), confirmed out-of-sample.
- Rejected factors keep their ID and a one-line reason; do not delete the row.
- When adding a factor, also record its hypothesis and decision rule in the
  pre-registration log before running the test.
