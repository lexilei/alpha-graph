# Adversarial audit — `scripts/qa_singles_iv_coverage.py` (2026-07-15)

Read-only review by an independent agent; every claim demonstrated on the
actual store; all headline numbers reproduced on a deliberately different
code path (idxmax dedup + merge pairing + sort/groupby ATM vs the script's
sort/drop_duplicates + pivot + idxmin).

## FINDINGS (ranked)

**F1 — Fixed-calendar denominator understates IPO / partial-download names
(MUST FIX before the 61-name rollout).** spot.parquet carries an identical
fixed 2,579-day calendar (2016-01-04→2026-04-07) for AAPL/AMD/AMZN — it is
calendar-based, not listing-based. The universe includes mid-window IPOs
(APP 2021-04, PLTR 2020-09): ~1,300 pre-IPO days would enter their
denominators, reporting a 100%-of-tradeable-life name at ~50% and falsely
failing the ≥80% bar — pulling the passing count below the ~55 minimum in
the conservative direction. Unproven on-target (those names not yet
downloaded); mechanism real. Fix: life-adjusted denominator.

**F2 — Coverage can't distinguish "bad data" from "unfinished download"
(AMZN 72.7%).** Two invisible causes: (a) post-2023 truncation (703 days,
expirations stop ~2023-07); (b) a genuine 5-day DTE hole 2021-09-13..17 —
store has only DTE {11,18,67,95,...} for those days; the 20211015 monthly
expiration partition is absent. Fix: report first/last quote date and
partition count per name.

**F3 — ±10% band narrower than the strike grid for low-priced names.**
AMD's single missing day (2016-03-10, spot ~$2.25, $0.50 grid = 11% steps;
nearest paired strikes ±11.1%) is legitimate but band-induced. Quantified:
across 2,579 AMD days only 9 fall in the 8-12% fragile zone, all at spot
< $3; above ~$5 it never bites. Understatement-direction only. No code
change; documented as a known floor.

**F4 — "Any one near pair marks the day usable" is a liveness floor,**
matching I2's minimal need, not a chain-richness measure (I1 skew wants
more). Documented, accepted.

**F5 — Cosmetic:** docstring promised a "monthly cross-section curve" that
isn't produced; `:.1%` display rounds AMD's 99.96% to "100.0%" (gate math
uses the true float).

## VERIFICATION

All five names reproduce exactly (AAPL/ADBE/AMAT 2579, AMD 2578 with the
2016-03-10 exclusion verified legitimate, AMZN 1876 with identical
per-year breakdown). Parity/split fix confirmed: AAPL pre-2020-08 and
AMZN pre-2022-06 fully covered where spot-file banding zeroed them.

## NON-ISSUES (probed, fine)

`created` is datetime64[ms] (true temporal sort; zero dup keys);
pivot+dropna yields a unique RangeIndex so groupby.idxmin alignment is
exact (0 mismatches vs brute force); ask=0 mids never selected at ATM in
the probe; datetime64-vs-Timestamp dict keys hash-equal; both-legs bid>0
semantics match the docstring; calendar DTE consistent with the pin.

## VERDICT

Methodology sound for its purpose; numbers correct and independently
reproduced; the historical bug class (lookahead, silent rails, alignment)
is absent. Resolve F1 (+ F2 diagnostics) before the full-universe gate.

## Disposition (same day)

F1/F2/F5 applied to the script: life-adjusted coverage (denominator
restricted to [first, last] option-quote date present per name — immune
to IPO timing and spot backfill by construction), per-name first/last
quote date + partition count columns, docstring/display corrected. F3/F4
documented in the docstring as accepted floors.
