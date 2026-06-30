# Factor Registry

Canonical, numbered list of factors. **IDs are permanent**: once assigned, a
number always means the same factor. New factors append the next free ID;
numbers are never reused or renumbered, even if a factor is rejected. Refer to
factors by ID everywhere else (logs, reports, commit messages).

Next free ID: **10**

Status legend: `candidate` (under evaluation) · `accepted` (in the model) ·
`rejected` (tested, no incremental signal — ID retained as a tombstone).

Prediction target for all factors: `fwd_return_21d` (next 21 trading-day return).

| ID | name | source | status | definition |
|----|------|--------|--------|------------|
| 1 | `cosine_similarity` | 10-K | candidate | TF-IDF (1–2gram, 10k vocab, EN stopwords) cosine between a firm's consecutive **same-type** 10-K filings; vectorizer refit per pair. High = little language change. |
| 2 | `event_score` | 8-K | candidate | Exponentially-weighted avg (λ=0.9/month, 12-month lookback) of per-filing 8-K item prior scores; each filing scored by its most extreme item (`max\|score\|`). |
| 3 | `event_count` | 8-K | candidate | Number of 8-K filings in the lookback window. |
| 4 | `regime_state` | HMM | candidate | 3-state Gaussian HMM label on market-level features (0 trending / 1 mean-reverting / 2 crisis). **Market-level — identical across the cross-section on a given date.** |
| 5 | `regime_prob` | HMM | candidate | Posterior probability of the current HMM state. **Market-level.** |
| 6 | `momentum_21d` | price | candidate | `close.pct_change(21)`. |
| 7 | `momentum_5d` | price | candidate | `close.pct_change(5)`. |
| 8 | `volatility_21d` | price | candidate | 21-day rolling std of daily returns × √252 (annualized realized vol). |
| 9 | `volume_zscore` | volume | candidate | `(volume − 63d mean) / 63d std`. |

## Known issues (carry into any evaluation)

- **1** — corpus is incomplete pre-backfill (~30% of tickers had ≥2 consecutive
  10-Ks before 2024 → ~40% panel coverage). Numbers are not trustworthy until
  the 10-K backfill + cosine recompute lands.
- **2, 3** — item prior scores are hand-coded, never calibrated against realized
  forward returns.
- **4, 5** — market-level: zero cross-sectional ranking content (confirmed —
  `factor_orthogonality.py` cannot score them; they vanish on rank-standardization).
  Useful only as conditioning/interaction variables, not as stock-pickers.

## Convention

- A factor enters as `candidate`. Promote to `accepted` only after it clears the
  pre-registered incremental-IC threshold over the current accepted set
  (`scripts/factor_orthogonality.py`), confirmed out-of-sample.
- Rejected factors keep their ID and a one-line reason; do not delete the row.
- When adding a factor, also record its hypothesis and decision rule in the
  pre-registration log before running the test.
