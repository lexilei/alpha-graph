# Alpha-Graph

NLP-driven equity signal generation via SEC filing analysis and machine learning.

## Results (24-month out-of-sample, Mar 2024 – Feb 2026)

| Strategy | Cumulative | Sharpe | Max DD | Win Rate |
|----------|-----------|--------|--------|----------|
| **ML Combiner (L/S)** | **+80.5%** | **1.77** | **-13.3%** | **71%** |
| Lazy Prices + HMM | -19.9% | -1.04 | -28.6% | 50% |
| Lazy Prices (baseline) | -25.3% | -1.49 | -28.1% | 37.5% |
| S&P 500 (long only) | +44.8% | 1.41 | -7.6% | — |

The ML combiner (walk-forward LightGBM on 8-K events + cosine similarity + momentum + regime features) transforms the strategy from losing money to Sharpe 1.77.

## Architecture

```
SEC EDGAR (10-K)  ──> TF-IDF Cosine Similarity ──> Lazy Prices Score ──┐
SEC EDGAR (8-K)   ──> Item-Type Event Scoring   ──> Event Score ───────┤
Market Data       ──> 3-State Gaussian HMM      ──> Regime State ─────┤──> Walk-Forward
Market Data       ──> Momentum / Vol / Volume    ──> Price Features ───┘    LightGBM
                                                                            │
LLM Agents (optional):                                                      v
  Filing Analyst ────┐                                                 Predicted Return
  News Synthesizer ──┤──> Research Coordinator                         (per ticker/month)
  Earnings Analyst ──┘    (confidence-weighted)
```

**Key insight**: The 8-K event signal (updated ~40x/year per ticker) subsumes the annual Lazy Prices signal. Feature importance: `event_score` >> `momentum_21d` >> `cosine_similarity` = 0.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env  # add SEC EDGAR credentials, Together AI key
```

## Usage

```bash
# Download data (102 S&P 500 tickers, 3 years)
python -m alpha_graph.data.filings --max-tickers 100 --years-back 3
python -m alpha_graph.data.market --max-tickers 100 --years-back 3

# Generate signals
python -m alpha_graph.signals.lazy_prices       # TF-IDF cosine similarity
python -m alpha_graph.signals.event_signal      # 8-K event scoring (NEW)
python -m alpha_graph.signals.regime            # HMM regime detection

# Train ML combiner (walk-forward, 12-month rolling window)
python -m alpha_graph.signals.ml_combiner --train --predict

# Run LLM agent pipeline (optional, needs Together AI key)
python -m alpha_graph.agents.pipeline --max-tickers 100

# Backtest
python -m alpha_graph.backtest.walk_forward     # Lazy Prices walk-forward
python -m alpha_graph.backtest.tearsheet        # HTML reports

# Paper trade (needs Alpaca account)
python -m alpha_graph.trading.executor --dry-run
```

## Project Structure

```
src/alpha_graph/
    config.py                    # Central configuration (.env)
    data/
        universe.py              # S&P 500 ticker management
        filings.py               # SEC EDGAR downloader (10-K/10-Q/8-K)
        transcripts.py           # Finnhub earnings call transcripts
        market.py                # yfinance price/return data
    signals/
        lazy_prices.py           # TF-IDF cosine similarity
        event_signal.py          # 8-K event scoring (25 item types)
        filing_changes.py        # LLM-based change detection
        regime.py                # 3-state Gaussian HMM + market breadth
        ml_combiner.py           # Walk-forward LightGBM signal combiner
    agents/
        filing_analyst.py        # Filing Analyst (risk + opportunity)
        earnings_analyst.py      # Earnings Call Analyst
        news_synthesizer.py      # News Synthesizer (8-K events)
        coordinator.py           # Research Coordinator (IC-optimized weights)
        pipeline.py              # LangGraph orchestration
    backtest/
        engine.py                # Signal-weighted portfolio, sector caps
        walk_forward.py          # 24-month walk-forward backtest
        combined.py              # Combined LLM + regime test
        tearsheet.py             # Quantstats HTML reports
    trading/
        executor.py              # Alpaca paper trading
report/
    alpha_graph_report.tex       # LaTeX technical report
    fig*.png                     # Performance figures
tests/                           # 34 unit tests
```

## Data

| Source | What | Count |
|--------|------|-------|
| SEC EDGAR | 10-K filings | 401 |
| SEC EDGAR | 10-Q filings | 935 |
| SEC EDGAR | 8-K filings | 4,146 |
| Yahoo Finance | Daily OHLCV | 70,688 rows |

## Known Limitations

1. **Momentum dependence**: The ML combiner's top features are `event_score` and `momentum_21d`. During momentum crashes (e.g., Jan 2025: -9.0% in one month), the long leg gets hit hard because it's overweight growth/momentum names.
2. **Survivorship bias**: Only current S&P 500 constituents. Delisted stocks excluded.
3. **24-month OOS only**: Sharpe 1.77 over 24 months is promising but not definitive.
4. **No short borrowing costs**: Assumes costless shorting (realistic for large-cap but understates drag by ~30-50 bps/year).

See [METHODOLOGY.md](METHODOLOGY.md) for full discussion. See [TODO.md](TODO.md) for improvement roadmap.

## Tests

```bash
pytest tests/ -v  # 34 tests
```
