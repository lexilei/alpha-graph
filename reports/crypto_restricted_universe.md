# Restricted-universe rerun: the 4-sleeve combo on Coinbase-perp bases

2026-07-22. The headline combo (`crypto_factor_scan_k1.md`) was built on the
full Binance USDT-M panel — a venue a US person cannot legally trade. This
rerun restricts the point-in-time top-100 liquidity universe to symbols whose
base asset has a Coinbase perpetual (153 products fetched live from the
public Coinbase products API on the run date; 144 map onto the Binance
panel). Script: `scripts/crypto_restricted_universe.py`; protocol, sleeves,
fees and construction identical to the full-universe run.

## Results (run 2026-07-22)

| sleeve | FULL universe SR (2023+) | RESTRICTED SR (2023+) |
|---|---:|---:|
| mom30s1 | 0.95 (1.11) | 0.94 (1.45) |
| K2 carry_7d | 1.05 (0.58) | 1.19 (0.42) |
| K11 amihud_30d | 0.68 (0.34) | 1.48 (0.64) |
| K12 tbr_7d | 1.93 (1.93) | 1.69 (1.45) |
| **equal-risk combo** | **2.07 (1.75)** | **2.37 (1.90)** |

Mean universe size drops 89 → 51 names/day. Daily combo return correlation
between the two versions: 0.47.

## Reading — bounds, not a headline

The tradeable-universe restriction does **not** degrade the strategy: every
sleeve stays strongly positive and the combo point estimate is higher. But
the restricted number carries an additional bias the full-universe number
does not, and it must not be quoted without this caveat:

- **Listing-date lookahead.** The universe is conditioned on *today's*
  Coinbase listing roster applied retroactively. Coinbase lists assets that
  grew large and survived, so the historical restricted universe is
  implicitly winner-conditioned. K11 (illiquidity premium) jumping
  0.68 → 1.48 is the signature of exactly this bias — historically illiquid
  names that are Coinbase-listed *today* are disproportionately names that
  later grew. The bias inflates the restricted estimate by an unknown
  amount; point-in-time Coinbase listing dates (obtainable after account
  opening, or from listing announcements) would remove it.
- The script docstring's three standing caveats apply verbatim: the INTX
  product list is the **upper bound** of US-retail eligibility (the
  US-eligible subset must be confirmed after account opening); prices and
  funding are still Binance proxies for what would execute on Coinbase; and
  the 5bp taker fee assumption reflects Binance's schedule, not Coinbase's.

Honest summary for any external use: *restricting to a US-tradeable
universe does not collapse the combo (point estimate 2.37/1.90 vs
2.07/1.75), but the restricted estimate is upward-biased by listing-date
lookahead; treat the pair as bounds and the live expectation as unchanged
(shaded to 0.8–1.2 per `crypto_combo_risk.md`).*
