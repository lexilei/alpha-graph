# TODO: Alpha-Graph Improvements

## High Priority

### Momentum Crash Protection
- [ ] Add momentum reversal detector: when momentum factor drawdown > 10% in trailing 21 days, reduce long exposure to 50%
- [ ] In CRISIS regime, auto-reduce both legs to 25% (currently model ignores regime for position sizing)
- [ ] Add stop-loss at portfolio level: if monthly L/S return < -5%, flatten all positions

### Signal Diversification (reduce momentum dependence)
- [ ] Add low-volatility factor: prefer low-vol stocks for longs during high-VIX periods
- [ ] Add quality factor features: ROE, debt/equity, earnings stability
- [ ] Add value factor: P/E, P/B as features to ML combiner (avoid crowded growth trades)
- [ ] Retrain ML combiner with expanded feature set and evaluate IC improvement

### Regime-Aware Position Sizing
- [ ] Use regime state to dynamically scale gross exposure (currently equal weight regardless of regime)
- [ ] TRENDING: 100% long, 25% short
- [ ] MEAN_REVERTING: 100% long, 100% short
- [ ] CRISIS: 25% long, 25% short
- [ ] Backtest regime-scaled portfolio vs equal-weight

## Medium Priority

### Data Expansion
- [ ] Expand universe from 94 → 500 tickers (full S&P 500)
- [ ] Add survivorship-bias-free historical constituents (EODHD ~$22/month)
- [ ] Upgrade Finnhub to paid tier for earnings transcripts → activate earnings_analyst agent
- [ ] Add 10-Q filings to Lazy Prices signal (quarterly updates instead of annual)

### Model Improvements
- [ ] Hyperparameter tune LightGBM via Optuna (currently hand-picked conservative params)
- [ ] Test XGBoost and CatBoost as alternatives
- [ ] Add rolling IC monitoring: if trailing 3-month IC < 0, switch to momentum-neutral mode
- [ ] Implement ensemble of multiple combiner models

### Portfolio Construction
- [ ] Optimize number of positions (currently 10/10, test 15/15 and 20/20)
- [ ] Add risk parity weighting (inverse-vol weighted positions)
- [ ] Implement sector-neutral constraint in walk-forward backtest
- [ ] Add short borrowing cost model (~30-50 bps annually for S&P 500)

## Low Priority

### Infrastructure
- [ ] Set up Alpaca paper trading with ML combiner signals
- [ ] Add daily signal monitoring dashboard (Streamlit)
- [ ] Implement intraday rebalancing around 8-K filing timestamps
- [ ] Add Slack/email alerts for regime changes and large signal moves

### Research
- [ ] Knowledge graph: extract supplier/customer relationships from 10-K filings
- [ ] GNN-based cross-firm signal propagation
- [ ] Test signal on non-US markets (UK, EU filings)
- [ ] Investigate DeepSeek-V3 vs GPT-4o-mini for filing analysis quality
