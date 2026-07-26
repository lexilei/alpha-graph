# Pre-registration — identity resolution fix (equity line)

Written **before** any code change, per postmortem rule 2 ("quantify the
dominant loss factor on paper and prove it controlled before committing").
Snapshot of record: `2026-07-25-v2` (hash-verified 2026-07-26).

## Why this exists

The equity line's factor panel was found to be survivorship-biased: 503
yfinance tickers = today's Wikipedia S&P 500 list, against 811 unique PIT
members over the same window. Rebuilding the panel from Sharadar requires a
date-aware symbol -> vendor-identity resolution that currently does not
work well enough, and the project's own QA already says so.

## Root cause (measured, not assumed)

`sharadar_qa.py` contains two resolvers with asymmetric capability:

| resolver | methods | result |
|---|---|---|
| `resolve_identity_intervals` (:435) | exact + reuse_suffix + related_ticker + approved_override | 814/828 intervals resolved (98.3%) |
| `_resolve_symbol_ids` (:859) | exact ticker only | 421/497 at 2011-04-30 (84.7%) |

The monthly membership gate runs on the weaker one. Its unresolved count
(76 at 2011-04, decaying to ~0 as re-keyed tickers age out of the index) is
penalised twice — excluded from the intersection and added to the
denominator: `421/(500+76) = 0.731`. That, not corrupt data, is the
71.601% worst-month Jaccard.

## Baseline (2026-07-25-v2, measured 2026-07-26)

| metric | now | gate |
|---|---|---|
| min month-end Jaccard | 71.601% | 99% |
| max `unresolved_reference` in a month | 76 | — |
| interval crosswalk unresolved | 14 of 828 (23,515 member-days, 1.26%) | 0 ambiguous |
| departed identity coverage | 94.915% | 98% |
| member-day price coverage | 98.738% | 99% |
| identity registry rows | 0 (coverage 0.000%) | 100% |
| known identity cases adjudicated | 0/14 | 14/14 |
| membership differences adjudicated | 0/135 | 135/135 |
| price jumps math-verified | 4,676 / 68,853 (6.8%) | 0 unexplained |
| SEP rows quarantined | 6,091 | 0 |

## Step 1 scope — ACTIONS-driven resolution ONLY

Add the ticker-change chain to identity resolution and remove the
capability asymmetry. Source material already on disk and unused for
identity: ACTIONS `tickerchangefrom`/`tickerchangeto` (13,424 pairs),
`acquisitionof`/`acquisitionby` (8,247), `mergerto`/`mergerfrom` (134),
`delisted` (19,196). Today ACTIONS is read only by the price-jump audit.

### Done means all of

1. **No capability asymmetry** — both resolvers share one implementation.
2. **`unresolved_reference` max across all months: 76 -> <= 5.**
3. **Interval crosswalk unresolved: 14 -> <= 3**, each residual named with
   its reason.
4. **No regression: all 814 currently-resolved intervals resolve to the
   SAME `vendor_id`.** A new matcher that silently re-maps an existing good
   match is a failure even if the headline numbers improve.
5. Min month-end Jaccard **>= 99%**, OR every residual difference is
   enumerated and shown to be a genuine reference-vs-vendor *membership*
   disagreement rather than an identity failure. Stating which of the two
   is required; "it went up" is not acceptance.
6. The 14 known cases are each reported as mechanically-resolved or
   needs-human-evidence. The split is the deliverable, not a pass.

### Explicitly OUT of scope for step 1

Price-jump verification (64,177); delisting returns; the listing/issuer
registry; the panel rebuild itself. Each gets its own step and its own bar.

## Falsification

If after the fix min Jaccard is still < 99% **and** the residual is not
attributable to genuine membership disagreement, then the reference
membership CSV itself is suspect and the panel plan must be re-examined
before any factor is re-run. That outcome gets written down, not worked
around.

## Open questions not settled by this document

- **Delisting returns.** ACTIONS `delisted` carries a `value` that looks
  like deal size in $M (Linde/Praxair 47,306.2), not per-share
  consideration. Including dead names without booking what happened to the
  money is only half a survivorship fix (Shumway 1997). Source unresolved.
- **`close` vs `closeadj`.** Switching makes every FACTORS.md number
  incomparable. Must be declared, not defaulted.
- **Forward-return convention.** FACTORS.md declares PIT + t+1; the old
  panel computed `pct_change(21).shift(-21)` = close(t+21)/close(t). To be
  checked against the IC evaluator, not assumed.
- **The 29 tombstones.** FACTORS.md counts them toward the multiple-testing
  N. Whether they still count after the panel changes alters the
  significance bar on the 6 survivors. A decision, not a default.

## Universe

S&P 500, decided 2026-07-26. Not for data-volume reasons (SEP is already
downloaded in full; narrowing saves only cheap compute) but because the PIT
membership file, the QA thresholds (`cross_section_min` 490 /
`cross_section_max` 515) and all 35 prior factor results are S&P-shaped.
Comparability is the reason.

## Clock

Licence `license_expires` 2026-08-25 — 30 days from writing.
