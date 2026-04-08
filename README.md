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

The one thing that does survive 14 years of out-of-sample evaluation is the **long-only top10** book (Method E in `backtest/improvements.py`):

| Metric | Value |
|---|---|
| Cumulative return (168 months) | **+3,732%** |
| Annualized return | **+29.7%** |
| Sharpe | **1.08** |
| Max drawdown | **-32.4%** |
| Win rate | **65%** |

After running a SPY-beta attribution (`backtest/attribution.py`) on Method E:

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

So the long-only book has an alpha t-stat above 4, near-zero market beta, and is not a disguised mega-cap basket (Mag 7 contributes only 4.3% of all picks; top-5 names are 8% combined). This is a defensible weak alpha as a stock-picking signal, **but it is not the long/short market-neutral strategy this project was originally built to deliver**, and a single +29.7% annualized number on a US equity long book over 2011–2026 needs to be weighed against the fact that SPY itself returned +13.7% annualized over the same window with a Sharpe of 0.99.

In other words: the "win" here is that the signal correctly identifies a quality/persistence basket on the long side. There is no working short side, no working market-neutral combination, and the gross outperformance over SPY is partly real alpha and partly a high-beta-like exposure that we've only partially stripped out.

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

The code is unchanged from when the strategy was producing the inflated numbers — this update is documentation only. The repo still contains the full pipeline:

- **Data layer** (`src/alpha_graph/data/`): SEC EDGAR downloader, yfinance market data, fundamentals, an LLM-extracted inter-company relationship graph.
- **Signals** (`src/alpha_graph/signals/`): Lazy Prices TF-IDF cosine similarity, 8-K item-type event scoring, Gaussian HMM regime detector, walk-forward LightGBM combiner, knowledge-graph spillover (disabled — hurt OOS).
- **Multi-agent LLM pipeline** (`src/alpha_graph/agents/`): LangGraph fan-out/fan-in over Filing Analyst, Earnings Analyst (inactive — no transcript data), News Synthesizer, Research Coordinator. This produces a per-ticker BUY/SELL/HOLD with confidence-weighted scoring, and was used for one snapshot run; it is not part of the 14-year backtest.
- **Backtest** (`src/alpha_graph/backtest/`): walk-forward engine, cost-sensitivity sweep, ten alternative method variants in `improvements.py`, beta attribution in `attribution.py`, permutation test in `permutation_test.py`, the four post-hoc and four structural extensions in `extensions.py`.
- **Trading** (`src/alpha_graph/trading/`): Alpaca paper-trading executor, daily pipeline, monitor dashboard. Wired to an "Anti-Momentum" signal that was the headline of the previous README; that signal's headline Sharpe (~1.91) was generated on a 23-month subset and **did not survive the permutation test on the full panel** — see `permutation_test_null.parquet`.
- **Tests** (`tests/`): 8 unit-test modules covering signal generation, portfolio construction, coordinator logic, and graph operations.

The project structure tree, dataset counts, and "how to run" commands are unchanged from earlier versions of this README; the only thing that changed today is that the result claims now match the cached data.

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
3. **Survivorship bias.** Universe is current S&P 500 constituents only. Companies that were removed from the index over the 14-year window are excluded, which mechanically inflates long-side returns.
4. **No short borrowing costs.** All numbers above assume costless shorting. Realistic borrow + market impact for the bottom-decile names would push the L/S Sharpe further negative.
5. **The previous "Anti-Momentum Sharpe 1.91" headline does not survive the permutation test on the full 14-year panel.** The earlier 23-month finding was inside a regime where momentum-reversal features happened to fit the post-Jan-2025 rotation; the null distribution from `permutation_test_null.parquet` (mean 0.016, std 0.22) shows that random feature shuffles produce Sharpes of similar magnitude often enough that the original number can't be defended.
6. **The earnings transcript signal was never tested** — Finnhub free tier doesn't include transcripts, so the Earnings Analyst agent is built but inactive. This is worth noting because earnings transcripts are the one major signal source that academic literature suggests would actually add value, and we never got to evaluate it.

See `METHODOLOGY.md` for the longer methodology writeup with the same updated numbers, and `report/alpha_graph_report.tex` for the full LaTeX writeup (also updated).
