# Replication-first scan — 2026-07-15

Three parallel agents downloaded and analyzed the published cross-sectional
factor evidence; every number below was either re-computed locally from the
downloaded raw files or read from the published tables (file + table cited).
Raw materials: `data/reference/{osap,hxz,decay_literature}/`. This report is
the tracked record; IDEAS.md carries the resulting scores.

## Sources

1. **Chen-Zimmermann Open Source Asset Pricing** — `SignalDoc.csv` (331
   signals, 212 predictors) + `PredictorLSretWide.csv` (monthly long-short
   returns 1926-2024, OP construction). Post-publication and 2011+ t-stats
   re-computed locally (verified: reproduces the agent's numbers exactly).
   Filter result: 146 buildable-now + 10 buildable-with-fetch, 28 excluded
   as already tested/queued, 28 not buildable (18 need IBES) →
   `data/reference/osap/shortlist.csv` (156 rows).
2. **Hou-Xue-Zhang, Replicating Anomalies (RFS 2020)** — full Table 3
   extracted from the published PDF. NYSE breakpoints + value-weighted is
   the published design closest to our S&P 500 panel. Headline: 65% of 452
   anomalies fail |t|>=1.96; **trading frictions: 102/106 fail (3.8%
   replicate)**. Category rates and per-anomaly verdicts in
   `data/reference/hxz/VERDICTS.md`.
3. **Decay literature** — McLean-Pontiff (JF 2016): OOS decay ~26%,
   post-publication ~58% (published headline; free PDFs are the earlier
   82-characteristic draft with 10%/35% — provenance flagged); persistence
   concentrates in small/illiquid/high-idio names (draft Table 8: idio
   +4.05 p<.001, size −1.49 p=.013, $vol −1.67 p<.01). Chen-Zimmermann
   (RAPS 2020): pure data-mining bias only ~12.3% (SE 1.7pp) of in-sample
   returns — the decay is real arbitrage, not mirage. Jacobs-Müller
   (JFE 2020): the US is the only market with reliable post-pub decay.

## What changed in IDEAS.md

**P-rule formalized** (排序 section): peer-reviewed floor P=3; post-pub
intact → 4; post-pub dead → ≤2; HXZ NYSE-VW fail → 1 (U1-conditional);
habitat penalty −1 for small/illiquid/high-idio-concentrated effects on
this panel, restored by U1. Panel calibration: of 5 HXZ-VW passers we
tested (C25/C26/C28/C29/C17) only C17 survives at the margin → "HXZ-VW
pass, untested here" defaults to P=2.

**Re-scores** (post-pub t / 2011+ t, recomputed):
- I1 iv_skew: skew1 1.20/1.20 → P 4→3 (5.4). **I2 cp_iv_spread becomes
  queue head** (analog CPVolSpread 3.25/3.13 — survives).
- I9 Amihud: OSAP 0.36 / 2011+ −0.83, HXZ NYSE-VW fail → P 2→1 (3.2).
- I13 turnover: OSAP 0.29/0.15, HXZ fail → P 2→1 (3.2).
- I17 O/S option volume: 0.40 / OV2 reversed → P 3→2 (3.6, bench).
- I20 momentum seasonality: registered same-day on HXZ in-sample 3.43,
  then killed by OSAP post-pub −0.20 → P 2→1 (3.6).
- I6 short interest: 3.98/3.10 post-pub STRENGTHENED → P=3 confirmed
  (4 base − 1 habitat); IO_ShortInterest variant is a microcap artifact.
- I5 breadth: 1.41/1.73 weak-positive → P=3 unchanged.
- I14 GKM: absent from both HXZ and OSAP — no independent tracking.

**New registrations**: I21 volume_trend (OSAP 5.08/3.84, the only modern
survivor in the trading class; P=2, 6.4), I22 div_seasonality
(DivYieldST 9.61/2.24 + DivSeason 2.11/2.87, large-cap habitat but tiny
means — gate-2 break-even required pre-registration; 5.0), I23
earnings_streak (4.28/4.71, must be incremental to C17's SUE; family
ruling pending; 3.6), I24 wq101_alphas (user source; no per-alpha
published stats, horizon mismatch, N-tax constraint — composite-single-
look or pre-registered ≤3 only; 1.8).

**Post-U1 shortlist** (Inbox): NetPayoutYield, XFIN, dNoa, Cop,
ShareIss5Y, Pda, Cei, roaq, SP, cfp, ChTax, Tax, Rdm, Ol, Noa — all
blocked by the fundamentals-family closure on the current universe,
queued for the U1 re-pin.

**Notable dead ends flagged** (don't re-try blind): BAB/BetaFP post-pub
reversed (−0.22), TrendFactor 0.48, MaxRet/IdioVol/Piotroski/betaVIX
dead post-pub; IVOL is genuinely absent in large caps (HXZ: fails even
EW). OSAP "IntMom" is Novy-Marx intermediate momentum (post-pub 1.05,
weak) — not Moskowitz-Grinblatt industry momentum, which OSAP does not
carry (I19's evidence is HXZ-only).

## Caveats

- OSAP post-pub series are OP-weighted (mostly EW) over the full CRSP
  universe including microcaps: a positive post-pub t does NOT guarantee
  large-cap survival. It ranks priors; it doesn't confirm anything.
- MP exact coefficients come from the draft PDF (published version
  paywalled; direction confirmed, magnitudes may differ slightly).
- EarningsSurprise (PEAD) full-universe 2011+ t = 1.03: modern PEAD is
  thin everywhere, consistent with C17's borderline +2.01.
