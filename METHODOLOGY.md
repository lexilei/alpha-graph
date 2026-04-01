# Methodology

## Economic Hypothesis

Information embedded in SEC filings propagates into stock prices with a measurable delay. Companies that substantially change their 10-K/10-Q language — particularly risk factors, litigation disclosures, and MD&A tone — tend to underperform in subsequent months. This "Lazy Prices" anomaly (Cohen, Malloy & Nguyen, 2020) documented portfolios earning ~188 bps/month alpha.

We extend this in three directions:

1. **8-K event-driven signals**: 8-K filings are filed on material events (auditor changes, M&A, officer departures) and provide ~40x higher signal frequency than annual 10-K filings. We score 25 different 8-K item types and exponentially weight recent events.

2. **ML signal combination**: A walk-forward LightGBM model combines filing similarity, event scores, market regime, and price features to predict forward 21-day returns. This replaces hardcoded heuristics with learned cross-sectional patterns.

3. **LLM-enhanced analysis**: Multi-agent LLM pipeline (Filing Analyst, News Synthesizer, Research Coordinator) extracts qualitative assessments that complement quantitative signals.

**Why this should work**: SEC filings are public but costly to process at scale. Systematic NLP extracts signals from the gap between what's filed and what's priced. The 8-K event signal captures information that arrives throughout the year, not just during annual filing season.

## Signal Generation

### 1. Lazy Prices (TF-IDF Cosine Similarity)

For consecutive 10-K filings $(d_{t-1}, d_t)$:
- TF-IDF vectors with bigrams, 10,000 term vocabulary, English stop words removed
- Cosine similarity: mean 0.896, std 0.204 across 299 filing pairs
- Cross-sectional quintile ranking: top quintile (least change) = +1, bottom = -1
- **Update frequency**: ~1x/year per ticker (major limitation)

### 2. 8-K Event Signal (NEW)

Scores 8-K filings by item type:
| Item | Event | Score |
|------|-------|-------|
| 1.01 | Material agreement | +0.30 |
| 1.02 | Agreement termination | -0.50 |
| 2.05 | Restructuring costs | -0.40 |
| 4.01 | Auditor change | -0.70 |
| 5.02 | Officer departure | -0.30 |

Exponentially-weighted average (λ=0.9 decay per month). 4,146 events across 102 tickers.
- **Update frequency**: ~40x/year per ticker
- **This is the dominant predictive feature** (importance = 6, vs cosine_similarity = 0)

### 3. HMM Market Regime

3-state Gaussian HMM on S&P 500 features (5-day return, 21-day volatility, volatility ratio).
Features standardized before fitting. BIC-based model selection between 3 and 4 states.

| Regime | Days | Fraction | Long Exposure | Short Exposure |
|--------|------|----------|---------------|----------------|
| Trending | 556 | 76.1% | 100% | 25% |
| Mean-Reverting | 90 | 12.3% | 100% | 100% |
| Crisis | 85 | 11.6% | 25% | 25% |

### 4. ML Signal Combiner (LightGBM)

Walk-forward training: 12-month rolling window, 1-month test, 5-day purge gap.

**Features**:
| Feature | Coverage | Importance |
|---------|----------|------------|
| event_score | 100% | 6 |
| event_count | 100% | 3 |
| momentum_21d | 97% | 2 |
| regime_state | 97% | 1 |
| momentum_5d | 99% | 1 |
| volume_zscore | 92% | 1 |
| cosine_similarity | 66% | 0 |
| volatility_21d | 97% | 0 |

**Walk-forward IC**: Mean 0.062, ICIR 0.61, positive in 18/24 months (75%).

### 5. Multi-Agent LLM Pipeline (Optional)

LangGraph fan-out/fan-in architecture with DeepSeek-V3:
- **Filing Analyst**: Balanced risk + opportunity prompt. Scores both risk (-1 to 0) and opportunity (0 to +1).
- **News Synthesizer**: 8-K event categorization and impact scoring.
- **Research Coordinator**: IC-optimized confidence-weighted signal combination.

## Portfolio Construction

- Monthly rebalance, top 10 long / bottom 10 short
- Signal-weighted positions (stronger signal = larger weight)
- Position cap: 5% per stock
- Sector cap: 30% per sector
- Transaction costs: 20 bps per rebalance (conservative for S&P 500)

## Results

### ML Combiner Long/Short (24-month OOS, Mar 2024 – Feb 2026)

| Metric | Value |
|--------|-------|
| Cumulative Return | **+80.5%** |
| Annualized Sharpe | **1.77** |
| Max Drawdown | -13.3% |
| Positive Months | 71% (17/24) |
| Avg Monthly Return | +2.61% |
| Best Month | +11.7% (Dec 2024) |
| Worst Month | -9.0% (Jan 2025) |

### Comparison

| Strategy | Cum Return | Sharpe | Max DD |
|----------|-----------|--------|--------|
| **ML Combiner** | **+80.5%** | **1.77** | **-13.3%** |
| Lazy Prices + HMM | -19.9% | -1.04 | -28.6% |
| Lazy Prices (baseline) | -25.3% | -1.49 | -28.1% |
| S&P 500 | +44.8% | 1.41 | -7.6% |

### Performance by Half-Year

| Period | Mean IC | Interpretation |
|--------|---------|----------------|
| H1 2024 | +0.055 | Momentum works, growth outperforms |
| H2 2024 | +0.065 | Same regime, consistent signal |
| H1 2025 | -0.014 | **Momentum crash** — signal fails |
| H2 2025 | +0.135 | Growth recovers, model works again |

## What Doesn't Work and Why

### 1. Jan 2025 Momentum Crash (-9.0%)

The model's long portfolio (APP, CVNA, AXON, BX, ANET) was concentrated in high-beta growth names because `event_score` and `momentum_21d` dominate feature importance. During the Jan 2025 risk-off episode:
- Long leg: -15.5% (ANET -25.8%, CCL -20.8%, AXON -19.0%)
- Short leg: -6.8% (defensive names held up: BAX +7.4%, APTV +0.6%)
- HMM oscillated between CRISIS and MEAN_REVERTING every day

**Root cause**: The model is implicitly a momentum + event factor. When momentum reverses, both signals point the wrong way.

### 2. Lazy Prices Signal is Subsumed

Cosine similarity has zero feature importance in the final model. The 8-K event signal captures the same information (companies with negative events also tend to rewrite filings) at 40x higher frequency. Lazy Prices alone cannot sustain a monthly-rebalanced portfolio.

### 3. Survivorship Bias

Only current S&P 500 constituents. Companies removed from the index (typically after underperformance) are excluded, biasing results upward for longs and downward for shorts.

## Improvement Roadmap

1. **Momentum crash protection**: Detect momentum factor drawdown > 10%, auto-reduce long exposure
2. **Factor diversification**: Add quality, value, low-vol features to reduce momentum dependence
3. **Regime-aware sizing**: Scale gross exposure by regime (currently ignored by ML combiner)
4. **Larger universe**: 500 tickers with survivorship-bias-free data
5. **Earnings transcripts**: Activate earnings_analyst agent (needs paid Finnhub)

See [TODO.md](TODO.md) for the full roadmap.

## Interview Questions

**Why should this work?**
SEC filings are public but costly to process at scale. The 8-K event signal captures material information (auditor changes, officer departures, restructuring) that the market underreacts to. The ML combiner learns cross-sectional patterns from a diverse feature set rather than relying on a single anomaly. ICIR of 0.61 over 24 OOS months provides statistical evidence of predictive power.

**How do you know it's not overfit?**
(1) Walk-forward validation with 5-day purge gap — each month is predicted by a model that never saw future data. (2) Conservative LightGBM hyperparameters (15 leaves, strong L1/L2 regularization). (3) The IC is positive in 75% of months, not just a few lucky periods. (4) Transaction costs modeled at 20 bps. (5) The worst month (-9.0%) and H1 2025 IC collapse are reported honestly.

**What breaks it?**
(1) Momentum crashes — the model is overweight momentum/growth, which fails during risk-off. (2) Regime dependence — 76% of the test period was trending, favoring long bias. A prolonged bear market would stress the short signal differently. (3) Signal crowding — as more firms adopt NLP filing analysis, the information edge erodes.
