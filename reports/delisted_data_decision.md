# Delisted-data sourcing decision

Decision date: 2026-07-13. Research only. No subscription has been purchased.

## Decision

Do not buy an annual data subscription. After the ingestion and blind-QA
harness is complete, approve one month of Sharadar SFA Full History Personal at
USD 79 as an R&D expense.

Amended 2026-07-25: the pre-purchase written-confirmation gate was removed by
user decision. The one-month term is accepted as-is under the published terms;
`--license-expires` and `purge` remain the operative controls.

This is not approval to renew. Renewal requires a new factor to pass the full
promotion gate after the recurring data fee is included. At USD 10,000 of NAV,
12 months of SFA costs USD 948, or 9.48% of NAV. Norgate Platinum costs USD
630/year, or 6.30% of NAV. Neither recurring fee is supported by the project's
current economics.

The pre-registered C15 tradeable backtest has already failed and its live path
is closed. The measured gross spread was about 1.74%/year, net Sharpe was about
zero, doubled-cost Sharpe was -0.27, and break-even cost was 4.21 bp/side versus
the required 6.0 bp/side. New data may be used to audit the research result and
to study new pre-registered factors. It must not be used to tune or rescue C15.
See `reports/gate2_c15_tradeable.md`.

## Verified product facts

| Product | Relevant entitlement | Current personal price | Operational fit | Decision |
|---|---|---:|---|---|
| Sharadar SEP Full History | SEP, TICKERS, ACTIONS, INDICATORS, METRICS; 21,000+ active and delisted tickers since 1998; no SP500 table | USD 49/month | Native REST/bulk API on macOS | Do not buy alone; too narrow for the planned factor sprint |
| Sharadar SFA Full History | SEP, TICKERS, ACTIONS, SP500, SF1, SF2, SF3, EVENTS, DAILY and related tables | USD 79/month | Native REST/bulk API on macOS | Conditional one-month R&D purchase |
| Norgate US Stocks Platinum | Delisted securities to 1990 and historical index constituents; stable numeric `assetid` | USD 346.50/6 months or USD 630/12 months | Python and NDU must run inside a Windows VM | Do not buy now; fallback only if Sharadar licensing or identity QA fails |
| EODHD | Low-cost delisted history, but no vendor-native PIT S&P 500 identity spine | USD 199/year | Native API | Not accepted as the production research source |

Official sources:

- Sharadar SEP product and live manifest:
  https://data.nasdaq.com/databases/SEP
  https://data.nasdaq.com/api/v3/datatable_collections/SEP?embed%5B%5D=plans
- Sharadar SFA product and live manifest:
  https://data.nasdaq.com/databases/SFA
  https://data.nasdaq.com/api/v3/datatable_collections/SFA?embed%5B%5D=plans
- Norgate packages and Python API:
  https://norgatedata.com/stockmarketpackages.php
  https://pypi.org/project/norgatedata/

SEP is keyed by `(ticker, date)`, SP500 by `(date, ticker, action)`, and
TICKERS by `(table, permaticker, ticker)`. `permaticker` is not present in the
fact tables and is not publicly documented as a CRSP PERMNO-equivalent
security identifier. It must be treated as an unverified vendor identifier
until share-class, ticker-reuse, and reorganization cases pass QA. Norgate's
API does expose an unchanging security-level `assetid`; the earlier report
understated this feature.

## What the current hole means

The current market panel contains 499 ticker lines and remains materially
survivor tilted. The exact missing-name denominator must not be hard-coded as
306/798. The source membership file contains ticker snapshots, not stable
security identities or company names. Depending on the date window and the
current rename compatibility map, the observed ticker-level denominator
changes. The procurement audit must calculate and persist its denominator only
after membership intervals have been resolved to vendor listing identities.

The direction of the bias is not known in advance:

- Omitting a negative terminal return overstates a long position but
  understates the gain on a short position.
- Omitting borrow recalls, locate failures, squeezes, and forced buy-ins can
  overstate a short strategy.
- Omitting an entire departed company can move either portfolio leg depending
  on its signal rank.
- Index removal, cash acquisition, stock acquisition, ticker change,
  bankruptcy, and exchange delisting are different events and require
  different exit handling.

All results must therefore show long and short legs separately and report exit
assumption sensitivity. No accessible candidate supplies CRSP-style `DLRET`.
A 0%/-30%/-100% terminal-return band may be applied only when a held security
has a verified performance delisting and no executable exit price. It must be
applied symmetrically to long and short holdings. Ordinary index removals and
corporate transactions must use their actual next-tradable or transaction
terms instead.

## License terms

Nasdaq Data Link terms require use to stop and supplied Data to be deleted when
the order expires; the general terms permit some non-reversible Derived Data.
Plan the month on the assumption that raw and normalized vendor files are
deleted at expiry: `--license-expires` pins the date into the snapshot manifest
and `purge` performs the deletion with a certificate.
Terms: https://data.nasdaq.com/terms

Norgate explicitly permits personal investment/trading, but its EULA requires
Data and Derived Data to be deleted when the subscription lapses, so it is not
a one-time-download workaround: https://norgatedata.com/subscribe/eula.php

## Blind acceptance gates

These checks are evaluated before any factor IC, spread, Sharpe, or P&L is
computed from the vendor data:

1. Preserve immutable raw Parquet files, API predicates, file sizes, hashes,
   vendor snapshot time, schema, and license expiry. Never overwrite
   `data/cache/market_data.parquet` during evaluation.
2. Create separate `listing_id` and `issuer_id` fields. CIK may support issuer
   matching but is not a listing key. Every membership and price row must map
   to exactly one active listing or enter quarantine.
3. Resolve at least 98% of the dynamically calculated departed-listing target.
   All remaining cases require an explicit reason; silent ticker fallback is
   prohibited.
4. Manually adjudicate 100% of the high-risk cases: FB/META, FI/FISV,
   IR/TT and the reused IR, MWV/WRK versus SW, DISH versus SATS, PX/LIN,
   DWDP/DD, GOOG/GOOGL, FOX/FOXA, NWS/NWSA, and DISCA/DISCK.
5. Vendor primary keys have zero duplicates, alias validity windows have zero
   unresolved overlaps, and all membership events resolve to one listing.
6. Identity-level month-end membership Jaccard versus the independent source
   is at least 0.99 through that source's final covered date. Differences over
   five trading days are individually adjudicated.
7. Overall member-day price coverage is at least 99%; every year is at least
   98%. There are zero unresolved non-positive prices, OHLC violations, or
   non-action single-day moves over 50%.
8. Preserve `closeunadj`, split-adjusted OHLC/close, `closeadj`, dividends, and
   actions as distinct fields. Do not use a total-return-adjusted price as an
   executable price.
9. Backfill departed-company SEC filings and `acceptance_ts` before any
   filing-factor full-universe claim. Price coverage alone does not fix text
   coverage.

## Execution sequence

1. Finish the bulk-download plan, checksum manifest, schema validation,
   identity quarantine, QA report, and license-expiry guard before activating
   a paid month.
2. Activate SFA for one month.
3. Download reference tables first, then the scoped historical S&P panel and
   research tables. Run identity and price QA without reading factor returns.
4. Backfill departed-company SEC filings and acceptance timestamps in
   parallel.
5. Freeze listing identity, price basis, corporate-action accounting, exit
   rules, borrow/locate behavior, and all costs.
6. Pre-register one full-universe data-integrity replication. Its result may
   update the research record but cannot reopen the original C15 live path.
7. A new factor reaches shadow or micro-live only after it independently
   clears multiple testing, fully loaded costs including recurring data fees,
   and a genuinely unused forward/paper period.

## Claims after integration

Passing these gates supports the narrow claim that the project has a
vendor-covered, identity-audited PIT price panel with disclosed residual
missingness. It does not by itself support CRSP-grade delisting returns,
complete filing-factor coverage, or a tradeable strategy. Those claims require
their separate evidence and gates.
