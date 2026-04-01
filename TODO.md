# TODO

## Now: Paper Trading Setup
- [ ] Set up Alpaca paper trading account
- [ ] Wire Anti-Momentum strategy signals → Alpaca executor
- [ ] Daily cron: fetch data → generate signals → submit orders
- [ ] Monitor dashboard (equity curve, positions, P&L)

## Next: Data
- [ ] Fetch more price/IV data for option_volitility project
- [ ] Download real Deribit option chains for vol_smile project
- [ ] Upgrade Finnhub → activate earnings_analyst agent

## Later: Improvements
- [ ] Expand universe to 500 tickers
- [ ] Add quality/value factors to ML combiner
- [ ] Longer backtest (need 5+ years of 8-K data)
