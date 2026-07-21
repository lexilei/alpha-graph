# Crypto perp cross-section: fee-net feasibility scan

2026-07-21. Question: does the simplest cross-sectional signal on Binance
USDT perpetuals clear realistic retail costs at daily rebalance? This is a
venue-feasibility check, not a factor claim — the full pre-committed grid is
reported, no variant selected.

## Data

Binance vision bucket (free, public): daily klines + funding for **787 USDT-M
perpetual symbols, 2020-01-01 .. 2026-06-30**, including 28 since-delisted
symbols (bucket listing retains them, so the panel is survivorship-free).
Downloader: `scripts/fetch_binance_perp.py`; panel in `data/raw/binance/`.
3 symbols with non-ASCII names failed the funding fetch (2026 meme listings,
present in klines; negligible). Funding files for klines-complete symbols:
2.42M funding events.

## Design (fixed before results were seen)

- Universe, point-in-time: top 100 by trailing 30d median quote volume,
  ≥30d history, stablecoin pairs excluded. Mean 89 members/day.
  2026-06 liquidity at the boundary: rank 10 ≈ $558M, rank 50 ≈ $60M,
  rank 100 ≈ $23M median daily volume — personal size is unconstrained.
- Signal grid: momentum `close[t-skip]/close[t-skip-lb] - 1`,
  lb ∈ {7, 30, 90} × skip ∈ {0, 1}.
- Portfolio: quintile long-short, equal weight, 1.0 gross per side, daily
  rebalance; signal at close t, position earns close t → close t+1.
- Costs: traded notional × fee, fee ∈ {0, 2bp maker, 5bp taker} (Binance
  regular tier, no BNB discount). Funding P&L accrued on held positions and
  included in all net rows (it is part of the strategy, not a cost knob).

## Results (`crypto_xs_mom_feasibility.csv`)

| lb | skip | 1-sided turn/day | gross SR | net maker SR | net taker SR | taker SR 2023– | funding ann |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 0.64 | 1.23 | 1.07 | 0.84 | 0.71 | +7.6% |
| 7 | 1 | 0.64 | 1.17 | 1.01 | 0.77 | 0.74 | +7.1% |
| 30 | 0 | 0.35 | 0.84 | 0.75 | 0.63 | 0.80 | +14.1% |
| 30 | 1 | 0.35 | 1.16 | 1.08 | 0.95 | 1.11 | +13.9% |
| 90 | 0 | 0.24 | 0.52 | 0.46 | 0.36 | 0.65 | +15.7% |
| 90 | 1 | 0.24 | 0.57 | 0.51 | 0.41 | 0.78 | +15.4% |

Ann. returns at 2.0 gross: 73%/68%/50%/68%/28%/31% gross across grid rows;
net taker 50%/45%/38%/56%/19%/23%. Ann. vol ≈ 55–60%, so max drawdowns at
this unscaled gross are −67% to −91%; any live version needs vol targeting
(Sharpe is the decision stat here, drawdown scales with chosen vol).

## Reading

1. **Feasibility passes.** Every cell of the grid is positive net of taker
   fees over 6.3 years, and the 2023-onward subsample (the competitive
   regime) does not kill it — mid-horizon momentum is *stronger* recently
   (lb30 skip1: 1.11 net taker). Contrast with the equity ledger, where 133
   looks produced 0 accepted factors: the same class of naive signal that is
   fully arbitraged in US equities still pays here after costs.
2. **Costs are the binding design constraint, as expected.** At lb7 the
   fee drag is 23%/yr at taker — maker execution or lower turnover is worth
   roughly as much as signal improvement. Mid-horizon (lb30) at 0.35
   turnover is the efficient point in this grid.
3. **Funding is a tailwind, not a cost** (+7–16%/yr): loser-quintile shorts
   collect positive funding. A separate funding/carry factor is the obvious
   next candidate.
4. This scan says the venue is worth building on. It does not say lb30
   skip1 is "the strategy" — that cell's edge over its neighbors is one
   look among six.

## Caveats

- Fill at daily close with no slippage beyond the fee. Top-100 perp spreads
  are ~1bp so this is minor, but a live check must measure it.
- Skew/beta not examined; L/S is dollar-neutral, not beta-neutral.
- Fee tiers assumed static at regular tier; BNB discount (−10%) ignored.
- Six looks logged. Any promoted variant must be re-registered under the
  ledger protocol with a pinned bar before live capital.

## Next

1. Vol-targeted version + beta-neutrality check.
2. Funding/carry as an independent factor; momentum–carry correlation.
3. Maker-execution turnover study (limit-at-close vs market-on-close).
4. Adapt the judge/registry conventions to a crypto panel and register
   candidates properly.
