# Alpha-Graph

NLP-driven equity signal generation from SEC filings (10-K, 10-Q, 8-K), tested on a 14-year cross-section of S&P 500 stocks.

## Honest summary: the long/short version of this strategy does not work

The original goal was a market-neutral long/short portfolio built on filing-text changes plus 8-K events. After running the backtest on the full 14-year panel (168 months, 499 tickers, ~1.66M ticker-months), the answer is unambiguous:

- **Top10/Bot10 long/short on the ML combiner signal**: cumulative **+90%** over 14 years (annualized **+4.7%**), Sharpe **0.32**, max drawdown **-58%**. That is worse than holding T-bills and dramatically worse than holding SPY.
- **Rank-weighted L/S** (the "obvious" generalization): Sharpe **-0.12**. Rank weighting assumes the signal is monotonic in forward return; it isn't.
- **Multi-factor combiner** (Lazy Prices + 12-1 Momentum + Low-Vol z-score average): Sharpe **-0.52**, cumulative **-59%**. Adding "diversifying" factors made it worse.
- **Pure 12-1 Momentum L/S** (sanity check): Sharpe **0.31**, cumulative **+86%**. So this universe isn't completely factor-hostile — momentum works at about the same level as the Lazy Prices L/S signal, which means our text-derived signal carries no useful information beyond what a trivial price-momentum factor already provides.

The 24-month "Sharpe 1.77 / +80.5%" headline that earlier versions of this README and the LaTeX report advertised was a cherry-picked window. On the actual cached `walk_forward_results.parquet`, the L/S strategy on the same period is **-44.9% cumulative, Sharpe -2.64**. The headline number was never reproducible.

## Why the L/S version fails: the signal is U-shaped, not monotonic

The diagnostic that broke the strategy is straightforward. We binned all ticker-months into deciles by Lazy Prices signal strength and measured the average forward 21-day return per decile:

| Decile | Signal | Avg Forward Return / Month |
|---|---|---|
| 9 (highest cosine sim, no filing changes) | strong long | **+2.10%** |
| 4–5 (middle, ambiguous) | neutral | **+1.19%** |
| 0 (lowest cosine sim, filing rewritten) | "strong short" | **+1.37%** |

The original Cohen, Malloy & Nguyen (2020) "Lazy Prices" thesis says the bottom decile should have the *worst* forward return — that's the alpha that motivates shorting them. In our 2011–2026 panel **the bottom decile beats the middle**. Companies that dramatically rewrite their filings do not underperform on average; they outperform the ambiguous middle.

This makes the long/short construction structurally broken:
- Rank-weighted L/S puts negative weight on the bottom decile, which has positive expected return → you are paying alpha to go *short* a positive-EV bucket.
- Top10/Bot10 has the same problem at smaller scale.
- Any "diversifying multi-factor" approach that assumes monotonicity dilutes the only working part of the signal (the long top decile) into the broken middle and short legs.

The short side of the original Lazy Prices paper does not replicate in our universe, and once you accept that, no combination of features, regime filters, or position-sizing tricks rescues the L/S strategy.

## What does survive: long-only top decile

The one thing that does survive 14 years of out-of-sample evaluation is the **long-only top10** book (Method E in `backtest/improvements.py`).

On the full (current-constituent) universe the numbers look strong:

| Metric | Full universe | PIT S&P 500 universe |
|---|---|---|
| Cumulative return (168 months) | +3,732% | **+888%** |
| Annualized return | +29.7% | **+17.8%** |
| Sharpe | 1.08 | **0.81** |
| Max drawdown | -32.4% | -41.8% |

**The PIT column is the honest number.** Forward survivorship bias (using 2026-constituent tickers in 2012, before they were in the index) inflated the annualized return by about 12 percentage points. Survivor bias from tickers removed before 2026 is not corrected here and remains an open caveat. See `report/u_shape_note.pdf` for details.

After running a SPY-beta attribution (`backtest/attribution.py`) on Method E (full universe):

| Beta-attribution metric | Value |
|---|---|
| Months in regression | 167 |
| α (annualized, arithmetic) | **+33.6%** |
| β vs SPY | **-0.19** |
| Alpha t-statistic | **+4.35** |
| Alpha p-value (two-sided) | **0.000023** |
| Residual Sharpe (α / σ_ε)·√12 | **+1.21** |
| Mag 7 share of all picks | **4.3%** |
| Top-5 ticker concentration | **8.0%** |

On the full-universe regression Method E has an alpha t-stat above 4 and near-zero market beta; on the PIT universe the FF5+MOM alpha drops to **+20.6% annualized** with **t = +3.00** (`backtest/ff_attribution.py` rerun on PIT predictions). Still statistically significant at 1%, but economically two-thirds of the unadjusted number.

`backtest/ff_attribution.py` (full universe) also rules out factor exposure: across CAPM, FF3, FF5, and FF5+MOM specifications, no factor loading has |t| > 2. The alpha does not come from disguised market, size, value, profitability, investment, or momentum exposure. `backtest/feature_stability.py` shows that across 156 walk-forward folds no single feature dominates — importance is bursty, with `cosine_similarity` selected in 68% of folds. Earlier claims that it was "subsumed" by the 8-K event score were based on a single-snapshot importance and are retracted. `backtest/realistic_slippage.py` shows Method E's Sharpe drops only from 1.08 to 1.04 under a per-ticker ADV-ranked cost model (avg 20.7 bps/month vs 10 bps flat), so the result is not fragile to cost assumptions. A market-neutral reframing (Method J: long top10 minus β·SPY, β estimated over trailing 24 months) delivers Sharpe 1.00 — confirming Method E was approximately market-neutral by construction (average hedging β = 0.10).

In other words: the "win" here is that the signal correctly identifies a quality/persistence basket on the long side. There is no working short side, no working market-neutral L/S combination, and the gross outperformance over SPY is partly real alpha (~18% ann on PIT) and partly forward survivorship bias.

## What the cached data actually says

The numbers above all come from re-reading `data/cache/improvements_results.parquet`, `method_e_attribution.parquet`, `walk_forward_results.parquet`, and `cost_sensitivity.parquet`. The full table from `improvements.py`:

| Method | Sharpe | Cumulative | Annualized | MaxDD | Win |
|---|---|---|---|---|---|
| Baseline: top10/bot10 (Lazy Prices L/S) | +0.32 | +90.4% | +4.7% | -58.4% | 54% |
| A. Rank-weighted L/S | -0.12 | -21.6% | -1.7% | -44.9% | 54% |
| B. Decile (top 10% / bot 10%) | +0.31 | +60.1% | +3.4% | -37.9% | 60% |
| C. Vol-targeted (10% ann vol) | +0.27 | +40.7% | +2.5% | -36.3% | 53% |
| D. Multi-factor (Lazy + Mom + LowVol, rank-wt) | -0.52 | -58.7% | -6.1% | -64.2% | 45% |
| **E. Long-only top10 (no short)** | **+1.08** | **+3,732%** | **+29.7%** | **-32.4%** | **65%** |
| F. Long top10 / Short MIDDLE10 (U-shape) | +0.61 | +341.6% | +11.2% | -34.1% | 55% |
| G. Pure 12-1 Momentum (no Lazy Prices) | +0.31 | +85.9% | +4.5% | -72.7% | 55% |
| H. 12-1 Mom + Low-Vol composite | -0.43 | -96.1% | -20.7% | -98.0% | 50% |
| I. Long Lazy Prices / Short low-Mom (cross-factor) | -0.04 | -43.8% | -4.0% | -72.6% | 52% |

Cost sensitivity for the L/S baseline (`cost_sensitivity.parquet`) — we currently assume 20 bps/month all-in:

| Cost (bps/mo) | Sharpe | Cumulative |
|---|---|---|
| 0 | +0.43 | +166% |
| 10 | +0.37 | +125% |
| 20 (assumed) | +0.32 | +90% |
| 50 | +0.15 | +15% |
| 75 | +0.02 | -24% |
| 100 | -0.12 | -50% |

So the L/S baseline has no margin: 50 bps/mo (well within real-world short-borrow + impact for many of the bottom-decile names) zeroes the Sharpe.

## What the code in this repo is

On 2026-04-15 the repo was cleaned up: the multi-agent LLM pipeline, paper-trading layer, knowledge-graph spillover, LLM filing-change detector, fundamentals feature, and transcripts pipeline were removed. They either produced no validated alpha or wired to retracted signals. The `pre-cleanup-2026-04-15` git tag preserves the prior state.

What remains:

- **Data layer** (`src/alpha_graph/data/`): SEC EDGAR filing downloader, yfinance market data.
- **Signals** (`src/alpha_graph/signals/`): Lazy Prices TF-IDF cosine similarity, 8-K item-type event scoring, Gaussian HMM regime detector, walk-forward LightGBM combiner.
- **Backtest** (`src/alpha_graph/backtest/`): walk-forward engine, cost-sensitivity sweep, ten alternative method variants in `improvements.py`, beta attribution in `attribution.py`, permutation test in `permutation_test.py`, four structural extensions in `extensions.py`.
- **Tests** (`tests/`): 5 unit-test modules covering signals, portfolio construction, ML combiner, and backtest engine.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env  # SEC EDGAR identity, optional Together AI key, optional Alpaca keys
```

## Reproducing the honest numbers

```bash
# Download data (one-time)
python -m alpha_graph.data.filings  --max-tickers 500 --years-back 14
python -m alpha_graph.data.market   --max-tickers 500 --years-back 14

# Build signals
python -m alpha_graph.signals.lazy_prices
python -m alpha_graph.signals.event_signal
python -m alpha_graph.signals.regime

# Train ML combiner walk-forward (this writes ml_combiner_predictions.parquet)
python -m alpha_graph.signals.ml_combiner --train

# Run all 10 portfolio variants on the 168-month panel
python -m alpha_graph.backtest.improvements

# Beta attribution for Method E (long-only top10)
python -m alpha_graph.backtest.attribution

# Cost sensitivity sweep for the L/S baseline
python -m alpha_graph.backtest.cost_sensitivity
```

## Tests

```bash
pytest tests/ -v
```

## Honest known limitations

1. **The L/S strategy doesn't work.** This is the main finding. The 14-year baseline Sharpe is 0.32, far below any deployment threshold, and the cost-sensitivity curve shows it goes to zero around 75 bps/month round-trip. Several "diversifying" multi-factor variants made things actively worse.
2. **The long-only book has alpha but isn't market-neutral.** Method E's t-stat-4.35 alpha is real, but the project was conceived as a long/short alpha capture, not a long-bias stock picker. Reframing it as a stock picker is honest but is not what was originally promised.
3. **Survivorship bias: partially corrected.** `backtest/pit_universe.py` applies a point-in-time S&P 500 membership filter (from the fja05680 GitHub dataset) which drops Method E's Sharpe from 1.08 to 0.81 and annualized return from +29.7% to +17.8%. This fixes the *forward-bias* component. The *survivor-bias* component (tickers removed from the index before 2026 and absent from our 499-ticker download) is not corrected and is the largest remaining caveat. The honest numbers to quote are the PIT-corrected ones.
4. **No short borrowing costs.** All numbers above assume costless shorting. Realistic borrow + market impact for the bottom-decile names would push the L/S Sharpe further negative.
5. **The previous "Anti-Momentum Sharpe 1.91" headline does not survive the permutation test on the full 14-year panel.** The earlier 23-month finding was inside a regime where momentum-reversal features happened to fit the post-Jan-2025 rotation; the null distribution from `permutation_test_null.parquet` (mean 0.016, std 0.22) shows that random feature shuffles produce Sharpes of similar magnitude often enough that the original number can't be defended.

See `METHODOLOGY.md` for the longer methodology writeup with the same updated numbers, and `report/alpha_graph_report.tex` for the full LaTeX writeup (also updated).
