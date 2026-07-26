# Dedupe — 2026-07-25-sfa

**Date** 2026-07-26 · **Freed** 2,865 MB (3,156 → 291 MB)

Two Sharadar snapshots of the same 2026-07-25 vendor data were on disk:
`2026-07-25-v2` (audited-module layout, hash-verified manifest) and
`2026-07-25-sfa` (standalone-script layout, unreadable by
`verify_snapshot_manifest` / `purge_snapshot_data`).

## Verification before deletion

`verify_snapshot_manifest` passed on `2026-07-25-v2`: schema v2, status
COMPLETE, 9 parts, every raw and staged file hash checked, inventories exact,
licence unexpired.

Equivalence of the duplicated tables was established directly, not assumed:

| check | tables | result |
|---|---|---|
| inner-CSV sha256 | SP500, ACTIONS, EVENTS | byte-identical |
| zip byte size | SEP, SF1, SF2, SF3, DAILY | delta = 0 |

## Deleted (8 tables, pure duplicates of v2)

SEP · DAILY · SF1 · SF3 · SF2 · EVENTS · ACTIONS · SP500

## Kept — NOT redundant

- **`SFP.zip`** (286 MB) — fund prices. Absent from v2 entirely.
- **`TICKERS.zip`** (5.0 MB) — v2's TICKERS is filtered to `[['table','SEP']]`
  (21,946 rows, SEP scope only). The sfa copy is the unfiltered superset:
  78,882 rows across SEP / SF1 / SF2 / SF3B / SFP scopes, **26,965 tickers
  that do not appear in v2 at all**. The SEP-scope rows are identical.
  Relevant to identity resolution (cf. `01f4e67`) and to any SF2/SF3 join
  that needs metadata for non-price-covered entities.
- `manifest.json` — provenance and licence trail for the two kept tables.

## Licence

`license_expires` 2026-08-25; expiry action is stop-use-and-delete. Both
snapshot directories are in scope on that date.
