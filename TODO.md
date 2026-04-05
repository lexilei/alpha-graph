# TODO

## Done: Paper Trading Setup
- [x] Set up Alpaca paper trading executor (`trading/executor.py`)
- [x] Wire Anti-Momentum strategy signals → Alpaca executor (`trading/signal_generator.py`)
- [x] Daily cron: fetch data → generate signals → submit orders (`trading/daily_pipeline.py`)
- [x] Monitor dashboard (equity curve, positions, P&L) (`trading/monitor.py`)

## Done: Feature Expansion
- [x] Add quality/value factors to ML combiner (`data/fundamentals.py`)
- [x] Expand universe default to 500 tickers (`config.py`)
- [x] Knowledge graph spillover signal (`signals/graph_signal.py`, `data/relationships.py`)

## Now: Data
- [ ] Upgrade Finnhub → activate earnings_analyst agent

## Next: Improvements
- [ ] Longer backtest (need 5+ years of 8-K data)
- [ ] Enable graph spillover features after more relationship data collected
- [ ] Run full 500-ticker backtest and compare to 100-ticker results
