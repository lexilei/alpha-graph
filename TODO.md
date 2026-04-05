# TODO

## Done: Paper Trading Setup
- [x] Set up Alpaca paper trading executor (`trading/executor.py`)
- [x] Wire Anti-Momentum strategy signals → Alpaca executor (`trading/signal_generator.py`)
- [x] Daily cron: fetch data → generate signals → submit orders (`trading/daily_pipeline.py`)
- [x] Monitor dashboard (equity curve, positions, P&L) (`trading/monitor.py`)

## Now: Data
- [ ] Upgrade Finnhub → activate earnings_analyst agent

## Next: Improvements
- [ ] Expand universe to 500 tickers
- [ ] Add quality/value factors to ML combiner
- [ ] Longer backtest (need 5+ years of 8-K data)
