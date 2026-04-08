# Methodology

This document describes what the system does, what we tried, and what the 14-year backtest actually says about the result. It is intentionally written in the past tense for everything that didn't work, because the previous version of this file overstated the findings on a cherry-picked 24-month window. Numbers here come from the cached parquet files in `data/cache/` (`improvements_results.parquet`, `method_e_attribution.parquet`, `cost_sensitivity.parquet`, `permutation_test_null.parquet`, `walk_forward_results.parquet`).

## Economic hypothesis (original)

We hypothesized that information embedded in SEC filings propagates into stock prices with a measurable delay, and that companies which substantially change their 10-K language — particularly risk factors, litigation disclosures, and MD&A tone — tend to underperform in subsequent months. This is the "Lazy Prices" anomaly of Cohen, Malloy & Nguyen (2020), who reported portfolios earning ~188 bps/month alpha by going long the most stable filers and short the heaviest rewriters.

We extended that idea in three directions:

1. **8-K event-driven signals**, scoring 25 different 8-K item types and exponentially weighting recent events. 8-Ks update ~40x/year per ticker, vs ~1x/year for 10-Ks, so the gain in signal frequency is large.
2. **ML signal combination** via a walk-forward LightGBM model that combines filing similarity, event scores, market regime, and price features to predict forward 21-day returns.
3. **LLM-enhanced analysis** via a multi-agent LangGraph pipeline (Filing Analyst, News Synthesizer, Research Coordinator) producing qualitative assessments to complement the quantitative signals.

## Signal generation (still in the codebase, unchanged)

### 1. Lazy Prices: TF-IDF cosine similarity

For consecutive 10-K filings $(d_{t-1}, d_t)$ we computed TF-IDF vectors with bigrams, 10,000-term vocabulary, English stop words removed, and took the cosine similarity. Mean 0.896, std 0.204 across 299 filing pairs. We then assigned cross-sectional quintile ranks: top quintile (least change) → +1, bottom quintile → -1.

Update frequency is ~1x/year per ticker, which is the major structural limitation of this signal.

### 2. 8-K event signal

Each 8-K item type has a hand-coded prior score reflecting expected market impact:

| Item | Event | Score |
|---|---|---|
| 1.01 | Material agreement | +0.30 |
| 1.02 | Agreement termination | -0.50 |
| 2.05 | Restructuring costs | -0.40 |
| 2.06 | Material impairment | -0.60 |
| 4.01 | Auditor change | -0.70 |
| 4.02 | Non-reliance on prior financials | -0.80 |
| 5.02 | Officer departure | -0.30 |

Per ticker we take an exponentially weighted average with λ=0.9 decay per month.

In the LightGBM combiner, this is the highest-importance feature (split count 6, vs 0 for cosine similarity), so the 8-K event signal effectively subsumes the annual Lazy Prices signal once both are available.

### 3. HMM market regime

A 3-state Gaussian HMM on standardized S&P 500 features (5-day return, 21-day vol, vol ratio, market breadth) with BIC-based selection between 3 and 4 states. State labels are assigned post-hoc by sorting on volatility:

| Regime | Days | Fraction | Long Exp | Short Exp |
|---|---|---|---|---|
| Trending | 556 | 76.1% | 100% | 25% |
| Mean-reverting | 90 | 12.3% | 100% | 100% |
| Crisis | 85 | 11.6% | 25% | 25% |

The dominant fact is that 76% of days were classified Trending. Any short-leg strategy was structurally disadvantaged for the entire test period.

### 4. ML signal combiner (LightGBM, walk-forward)

Walk-forward training with a 12-month rolling window, 1-month test, 5-day purge gap. Hyperparameters are conservative (15 leaves, learning rate 0.05, L1=0.1, L2=1.0, deterministic=True, n_jobs=1 for bit-reproducible fits).

Feature panel:

| Feature | Coverage | LightGBM importance (split count) |
|---|---|---|
| event_score | 100% | 6 |
| event_count | 100% | 3 |
| momentum_21d | 97% | 2 |
| regime_state | 97% | 1 |
| momentum_5d | 99% | 1 |
| volume_zscore | 92% | 1 |
| cosine_similarity | 66% | 0 |
| volatility_21d | 97% | 0 |

The model is trained on the full 14-year cross-section (1.66M ticker-months, 499 tickers, 168 monthly walk-forward folds). Predictions are saved to `ml_combiner_predictions.parquet`.

### 5. Multi-agent LLM pipeline

LangGraph fan-out/fan-in over three agents (Filing Analyst, News Synthesizer, Research Coordinator) using DeepSeek-V3 via Together AI. The Earnings Analyst is implemented but inactive (no transcript data on Finnhub free tier).

The pipeline was used to produce a one-period snapshot signal in `pipeline_signals.parquet`. It is not part of the 14-year backtest in `improvements.py`, which uses only the LightGBM combiner output. There is no longitudinal evaluation of the LLM pipeline because we never accumulated enough monthly snapshots.

## Portfolio construction

- Monthly rebalance.
- Top 10 long, bottom 10 short (or rank-weighted, vol-targeted, sector-neutral, etc. — see `improvements.py`).
- Dollar-neutral by construction in the L/S variants; long-only book has gross exposure 1.0 in Method E.
- Position cap 5% per stock, sector cap 30% per sector.
- Transaction costs: 20 bps per monthly rebalance flat (cost sensitivity below).

## Results

### The 14-year results that matter

All numbers below are from `improvements_results.parquet`, computed on the full 168-month / 1.66M-row panel. Costs are 20 bps/month except for Method E which uses 10 bps/month (long-only has no short leg).

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

### What this means

- **The L/S baseline Sharpe is 0.32.** That's worse than holding T-bills, with a 58% max drawdown. The strategy this project was originally built to deliver does not work.
- **Pure 12-1 momentum gives Sharpe 0.31** in this universe. That is essentially the same number as our text-derived L/S signal, which means our 14 years of work, ~5,400 SEC filings, and an ML combiner produce a long/short signal that is statistically indistinguishable from a one-line price-momentum factor anyone could compute in five minutes.
- **All multi-factor combinations are worse, not better.** Method D (Lazy Prices + 12-1 Momentum + Low-Vol z-score average) is Sharpe -0.52, cumulative -59%. Method H (Mom + LowVol) is -0.43. Method I (cross-factor hedge) is -0.04. The intuition that combining "diversifying" factors should improve Sharpe is wrong here, because the Lazy Prices L/S signal carries no information beyond what momentum already provides, so averaging it in only adds noise.
- **Method E (long-only top10) survives**, Sharpe 1.08, +3,732% over 14 years. But it isn't a long/short strategy and isn't what the project was supposed to deliver.

### Why the L/S strategy is structurally broken: the U-shape

We binned all ticker-months into deciles by Lazy Prices signal strength and looked at the average forward 21-day return per decile. The result:

| Decile | Signal | Avg forward return / month |
|---|---|---|
| 9 (highest cosine sim — no filing changes) | strong long | **+2.10%** |
| 4–5 (middle — ambiguous) | neutral | **+1.19%** |
| 0 (lowest cosine sim — filings rewritten) | "strong short" | **+1.37%** |

The original Lazy Prices paper assumes the bottom decile underperforms; that's the alpha that motivates shorting it. **In our 2011–2026 panel the bottom decile slightly outperforms the middle.** The signal is U-shaped, not monotonic.

The implications follow mechanically:
- **Rank-weighted L/S** (Method A) puts negative weight on the bottom decile, which has positive expected return → you are paying alpha to short a positive-EV bucket. Sharpe -0.12, exactly as predicted.
- **Top10/Bot10** has the same problem at smaller scale. The bot10 has slightly positive expected return; shorting it adds noise without alpha. Sharpe +0.32 — the entire L/S Sharpe is coming from the long top10, and the short side is dragging it down.
- **Long top10 / short middle10** (Method F, "exploit the U-shape") is Sharpe +0.61, almost double the standard top10/bot10. This confirms the U-shape diagnosis: shorting the actually-bad bucket (the middle) instead of the rewriter bucket helps. But "long top10 / short middle10" is not really a defensible production strategy — there's no economic reason to short the middle decile, the trade is harder to justify ex ante, and Sharpe 0.61 is still well below any deployment threshold.
- **Long-only top10** (Method E) is Sharpe 1.08 because it drops the broken short side entirely. This is just acknowledging that the long signal works and the short signal doesn't.

The short side of the original Cohen/Malloy/Nguyen finding does not replicate in our universe. Once you accept that, no combination of features, regime filters, sector neutrality, vol targeting, or momentum hedging makes the L/S strategy work. We tried all of those, and they're documented in `improvements.py` and `extensions.py`.

### Method E beta attribution

Method E (long-only top10) is the only thing with a Sharpe above 1, so we beta-adjusted it against SPY in `attribution.py` to check whether the alpha is real or disguised market exposure.

| Metric | Value |
|---|---|
| Months in regression | 167 |
| Method E raw Sharpe | +1.08 |
| Method E annualized return | +29.7% |
| SPY raw Sharpe | +0.99 |
| SPY annualized return | +13.7% |
| α (annualized, arithmetic) | **+33.6%** |
| β vs SPY | **-0.19** |
| R² | 0.009 |
| α t-statistic | **+4.35** |
| α p-value (two-sided) | **0.000023** |
| Residual std (monthly) | 6.4% |
| Residual Sharpe (α/σ_ε)·√12 | **+1.21** |

Beta is essentially zero (-0.19), R² is essentially zero (0.9%), and the alpha t-stat is 4.35 (p = 2.3e-5). The alpha is statistically distinguishable from market exposure, and the residual Sharpe of 1.21 actually slightly *exceeds* the raw Sharpe — so beta-adjusting strengthens rather than weakens the result.

Holdings concentration check:

| Metric | Value |
|---|---|
| Total ticker-month picks | 1,680 |
| Months covered | 168 |
| Unique tickers selected | 392 |
| Mag 7 share of all picks | **4.3%** |
| Top 5 ticker concentration | **8.0%** |
| Top 10 ticker concentration | ~14% |

The top picks are CVNA, BLDR, TSLA, NVDA, AXON, AMD, SMCI, BBY, APP, DXCM. Mag 7 contributes only 4.3% of all picks (versus the >30% threshold that would suggest a disguised mega-cap basket). The basket is reasonably diversified.

So Method E is a defensible weak alpha as a long-side stock picker. The caveats are:
- It is not what the project was built to deliver (long/short market-neutral was the goal).
- Survivorship bias is unmeasured. The universe is current S&P 500 constituents; companies removed during 2011–2026 are excluded, which inflates long-side returns by an unknown amount.
- The annualized 29.7% has to be compared honestly against SPY's 13.7% over the same window — the spread is real (16 percentage points), but a single 14-year long book of US equities outperforming SPY is not by itself proof of alpha; survivorship plus a quality/momentum tilt could explain a large fraction of it.

### Cost sensitivity for the L/S baseline

From `cost_sensitivity.parquet` — the L/S top10/bot10 strategy at varying transaction cost assumptions:

| Cost (bps/mo) | Sharpe | Cumulative | Max DD |
|---|---|---|---|
| 0 | +0.43 | +166% | -54% |
| 5 | +0.40 | +145% | -55% |
| 10 | +0.37 | +125% | -56% |
| 15 | +0.35 | +107% | -57% |
| **20 (assumed in baseline)** | **+0.32** | **+90%** | **-58%** |
| 30 | +0.26 | +61% | -61% |
| 50 | +0.15 | +15% | -65% |
| 75 | +0.02 | -24% | -71% |
| 100 | -0.12 | -50% | -77% |

Breakeven is ~75 bps/month (Sharpe goes to zero). 20 bps/month is on the optimistic end of plausible round-trip costs for monthly rebalancing of bottom-decile names that include illiquid and hard-to-borrow stocks; realistic costs would push the L/S Sharpe well below the headline 0.32.

### What happened to the previous "Sharpe 1.91 / Anti-Momentum" claim

An earlier version of this document and the LaTeX report headlined an "Anti-Momentum Features" extension at Sharpe 1.91 over a 23-month evaluation window. We ran a permutation test (`permutation_test.py`, results in `permutation_test_null.parquet`):

- Null distribution mean: +0.016
- Null distribution std: 0.22
- Null range: [-0.37, +0.47]
- Number of permutations: 50

The null distribution was generated by jointly shuffling the two anti-momentum features (`dist_52w_high`, `reversal_5d`) within each month and re-running the same walk-forward training. Random feature shuffles produce Sharpes in roughly [-0.37, +0.47]. The real number from the headline run was much higher than the null max — but on the **23-month window**, not the 14-year panel, and a 23-month window in 2024-2025 happens to be exactly the period when momentum reversal worked. On the full panel, anti-momentum features add no detectable signal.

The "Sharpe 1.91" number is not retracted in the codebase yet (the extension still runs and still produces it on the 23-month subset), but it is no longer claimed as a real result, and the production wiring (`trading/signal_generator.py`, `trading/executor.py`) that calls it "Anti-Momentum" should be understood as a research artifact, not a deployable strategy.

## What we are not claiming anymore

To make sure this document doesn't drift back into hype:

1. **There is no Sharpe 1.77 / +80.5% ML combiner result.** That number was from a 24-month cherry-pick, was inconsistent with the cached `walk_forward_results.parquet` for the same period (which shows -44.9%, Sharpe -2.64), and was never the result of the 14-year walk-forward.
2. **There is no Sharpe 1.91 / Anti-Momentum result.** That number was from a 23-month evaluation window and does not survive the permutation test on the full panel.
3. **There is no working long/short strategy** built from filing-text features in this codebase. The L/S baseline is Sharpe 0.32 over 14 years and breaks down completely under realistic short-side costs.
4. **There is no demonstration of LLM-derived alpha.** The multi-agent pipeline produces sensible per-ticker scores, but it was only run as a one-period snapshot (`pipeline_signals.parquet`), and we never accumulated enough monthly snapshots to run a longitudinal evaluation.
5. **There is no demonstrated edge from the knowledge graph spillover signal.** It is implemented in `signals/graph_signal.py` and `data/relationships.py`, but it hurt OOS returns and is disabled in the ML combiner feature list.

## What we are claiming

1. **The long-only top10 portfolio (Method E) delivers a defensible alpha** of ~+33.6% annualized vs SPY with t-stat 4.35, near-zero beta, and not driven by mega-cap concentration. This is a stock-picking result, not a market-neutral result, and it has survivorship bias as an unmeasured caveat.
2. **The Lazy Prices L/S signal is U-shaped, not monotonic, in our 2011–2026 panel.** The bottom decile slightly outperforms the middle. This is the structural reason every L/S variant we tried failed, and it's a useful finding in its own right because it means the original Cohen/Malloy/Nguyen short side does not replicate in this universe.
3. **The 8-K event score subsumes the 10-K filing similarity signal** when both are available (zero feature importance for cosine_similarity in the LightGBM model). If anyone builds on this, the cosine similarity signal can be dropped.

## Improvement ideas (recorded so they're not lost — not a TODO)

These are deliberately not in `TODO.md` (which has been deleted), because the user has instructed that the path forward is not to chase a better Sharpe but to be on the same page about what currently exists. They are recorded here as honest follow-up directions for whoever picks this up later.

- **Survivorship-bias-free universe.** The biggest unmeasured caveat for Method E. Backfilling the historical S&P 500 membership would cost effort but would tell us whether the +29.7% annualized number is real or partly survivorship.
- **Longer Lazy Prices history.** The current panel is 2011–2026 because that's how far back yfinance + edgartools easily go for our 499 tickers; pushing back to 2003 would test whether the U-shape we observe is stable or specific to the post-GFC era.
- **Earnings transcript signal.** Implemented but inactive. Academic literature suggests this is the most likely source of additional alpha, and we never tested it because Finnhub free tier doesn't include transcripts.
- **Don't promise a working long/short.** Future iterations should be honest from the start that the long-only book is what works in this universe. Building a long/short framing on top of a signal that is U-shaped is what got us into this situation.
