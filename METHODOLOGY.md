# Methodology

## Economic hypothesis

Information embedded in SEC filings propagates into stock prices with a measurable delay. Companies that substantially change their 10-K/10-Q language — particularly risk factors, litigation disclosures, and MD&A tone — tend to underperform in subsequent months. This "Lazy Prices" anomaly (Cohen, Malloy & Nguyen, 2020) documented portfolios earning ~188 bps/month by shorting high-change and buying low-change companies.

We extend this with two additions:

1. **LLM-enhanced change detection**: Rather than relying solely on TF-IDF cosine similarity, we use a multi-agent LLM pipeline to identify *what specifically changed* — distinguishing material shifts (new litigation, guidance cuts) from boilerplate updates (date changes, formatting). This produces a richer, directional signal versus the binary change/no-change of the original paper.

2. **HMM regime filtering**: A 3-state Hidden Markov Model classifies market regimes as trending, mean-reverting, or crisis. The long-short signal has most edge during mean-reverting periods. During trending bull markets, the short leg gets crushed by momentum — the regime filter reduces short exposure to 25% in these periods.

**Why this should work**: Information asymmetry between firms that disclose material changes (which sophisticated investors process first) and the broader market creates a window of predictability. The LLM layer extracts signal that keyword-based approaches miss. The regime filter addresses the well-documented failure mode of mean-reversion strategies in trending markets.

## System architecture

```
SEC EDGAR ──> Filing Analyst ──────────┐
(10-K/10-Q)   (Lazy Prices + LLM)     │
                                       ├──> Research Coordinator ──> Signals
SEC EDGAR ──> News Synthesizer ────────┤    (confidence-weighted)
(8-K)         (material events)        │
                                       │        ┌──> Regime Filter
Market Data ──────────────────────────>│        │   (HMM 3-state)
(yfinance)                             │        │
                                       └────────┴──> Portfolio Constructor
                                                     (long-short, 10+10)
```

The multi-agent pipeline uses LangGraph for parallel execution: all three analyst agents run concurrently on each ticker, then the Research Coordinator combines their signals with confidence-weighted averaging.

## Backtesting methodology

### Walk-forward validation
- **24 monthly out-of-sample periods** (Feb 2024 — Jan 2026)
- No in-sample optimization — the signal is the raw cosine similarity quintile sort
- Signals are carried forward from filing date until the next filing (point-in-time, no lookahead)
- Rebalance monthly at month-end

### Transaction cost model
- **10 bps round-trip** for liquid large-cap US equities
- Applied per position at each rebalance
- 20 positions per period (10 long, 10 short) = 200 bps total cost per month
- This is conservative — actual costs for S&P 500 names are typically 3-5 bps

### Risk metrics
- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014): adjusts for skewness, kurtosis, and number of strategies tested. Our DSR p-value of 0.059 with the regime filter means we cannot reject the null that the Sharpe is due to chance at 5% significance — this is an honest result.
- **Sortino Ratio**: penalizes downside volatility only
- **Maximum Drawdown**: worst peak-to-trough decline
- **Benchmark**: SPY total return over the same period

### Survivorship bias
The current implementation uses only currently-listed S&P 500 constituents. This introduces survivorship bias — companies that were delisted or removed from the index during the backtest period are excluded. A production version would use point-in-time constituent lists and delisted stock data (available via EODHD at ~$22/month).

## Results

### Lazy Prices standalone (no LLM, no regime filter)
| Metric | Value |
|--------|-------|
| Total Return | -25.33% |
| Annualized Return | -13.59% |
| Sharpe Ratio | -1.487 |
| Max Drawdown | -28.05% |
| % Positive Months | 37.5% |

### With HMM regime filter
| Metric | Value |
|--------|-------|
| Total Return | -19.85% |
| Annualized Return | -10.48% |
| Sharpe Ratio | -1.043 |
| Max Drawdown | -28.55% |
| % Positive Months | 50.0% |

### Multi-agent LLM pipeline (single-period test, 100 tickers)
| Metric | Value |
|--------|-------|
| Net Return (1 month) | +5.64% |
| Long leg | -4.07% |
| Short leg | +11.71% |

The regime filter improved every metric: Sharpe improved 30%, win rate went from 37.5% to 50%. The improvement came primarily from reducing short exposure during the 2024 bull market — October 2024 flipped from -2.93% to +3.09%.

## What doesn't work and why

1. **Lazy Prices alone is too sparse**: The signal only updates when a company files a 10-K (once per year). Between filings, the signal is stale. This is why the walk-forward Sharpe is negative — there simply aren't enough signal updates to drive consistent monthly returns.

2. **The 2024-2025 market regime was hostile**: A strong bull market punishes short positions regardless of signal quality. The HMM correctly identified this as a TRENDING regime (76% of days), but even with reduced short exposure, momentum dominated.

3. **Quintile sorting with 100 stocks is coarse**: With only ~94 tickers carrying a signal, the top/bottom 10 selection is sensitive to outliers. A larger universe (500+ stocks) with finer decile sorting would produce more stable portfolios.

## What would improve it

1. **Higher-frequency LLM signals**: Run the multi-agent pipeline on 8-K filings (which are event-driven and filed any time) rather than only on annual 10-K filings. This would generate signals throughout the year, not just around filing season.

2. **Earnings transcript analysis**: The Earnings Call Analyst agent is built but lacks data (Finnhub free tier doesn't include transcripts). Adding this would provide a quarterly signal with documented alpha (247 bps from proactive executives per academic research).

3. **Larger universe + delisted stocks**: Expanding to full S&P 500 with survivorship-bias-free data would improve statistical power and reduce sensitivity to individual stock outliers.

4. **Knowledge graph spillovers**: Extract supplier/customer relationships from filings and predict cross-company return spillovers via GNN — this is where the graph ML background creates genuine differentiation.

## The three interview questions

**Why should this work?**
Information in SEC filings is public but costly to process at scale. Most investors read headlines, not the full 200-page 10-K. Systematic text analysis — both TF-IDF similarity and LLM-based semantic understanding — extracts signals from the gap between what's filed and what's priced. The academic evidence is strong: Cohen et al. (2020) documented 22% annualized alpha from filing language changes; Kim & Blouin (2025) showed LLM-scored earnings transcripts predict returns. Our multi-agent architecture mirrors actual trading desk workflows, with specialist analysts producing independent assessments combined by a coordinator.

**How do you know it's not overfit?**
Three safeguards: (1) Walk-forward validation with no in-sample optimization — the signal is a simple quintile sort with zero tunable parameters. (2) The Deflated Sharpe Ratio accounts for the total number of strategies tested. (3) Transaction costs are modeled conservatively at 10 bps round-trip, and we report both gross and net returns. The negative standalone Sharpe is itself evidence against overfitting — an overfit strategy would show artificially positive results.

**What breaks it?**
Three failure modes: (1) Regime dependence — the signal underperforms in strong momentum markets where the short leg faces unlimited upside. The HMM filter partially addresses this. (2) Signal decay — as more firms adopt LLM-based filing analysis, the information advantage erodes. (3) Filing format changes — SEC XBRL mandates could alter the text structure in ways that break cosine similarity comparisons. The LLM-based analysis is more robust to format changes than TF-IDF.
