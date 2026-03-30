# alpha-graph

Multi-agent LLM pipeline for quantitative trading signals, built on SEC filings, earnings calls, and material event analysis.

## Strategy

Based on the "Lazy Prices" anomaly (Cohen, Malloy & Nguyen 2020) — companies that substantially change their 10-K/10-Q language tend to underperform (~188 bps/month alpha). Extended with LLM-based analysis of earnings call communication patterns (247 bps alpha for proactive executives) and 8-K material event detection.

**Architecture**: 4 specialized LangGraph agents process financial documents in parallel and produce a confidence-weighted BUY/SELL/HOLD recommendation:

```
START ──> Filing Analyst ──────┐
     ──> Earnings Analyst ─────┤──> Research Coordinator ──> BUY/SELL/HOLD
     ──> News Synthesizer ─────┘
```

## Setup

```bash
git clone https://github.com/lexilei/alpha-graph.git
cd alpha-graph
pip install -e ".[dev]"

cp .env.example .env
# Edit .env with your credentials
```

### Required API keys

| Service | Purpose | Cost |
|---------|---------|------|
| SEC EDGAR | 10-K/10-Q/8-K filings | Free (name + email) |
| Finnhub | Earnings transcripts | Free tier (60 req/min) |
| OpenAI | GPT-4o-mini for agent analysis | ~$20-40/month |

## Usage

```bash
# 1. Download data
python -m alpha_graph.data.filings --tickers AAPL MSFT GOOGL --years-back 3
python -m alpha_graph.data.transcripts --tickers AAPL MSFT GOOGL
python -m alpha_graph.data.market --tickers AAPL MSFT GOOGL

# 2. Generate signals
python -m alpha_graph.signals.lazy_prices          # Cosine similarity signal
python -m alpha_graph.signals.filing_changes       # LLM change detection
python -m alpha_graph.signals.regime               # HMM regime detection

# 3. Run multi-agent pipeline
python -m alpha_graph.agents.pipeline --max-tickers 100

# 4. Backtest
python -m alpha_graph.backtest.walk_forward        # Walk-forward (24 months OOS)
python -m alpha_graph.backtest.combined            # Combined LLM + regime
python -m alpha_graph.backtest.tearsheet           # Generate HTML reports

# 5. Paper trading (requires Alpaca account)
python -m alpha_graph.trading.executor --dry-run   # Simulate trades
python -m alpha_graph.trading.executor --live      # Submit paper trades
```

## Project structure

```
src/alpha_graph/
    config.py                  # Central configuration (.env)
    data/
        universe.py            # S&P 500 ticker management
        filings.py             # SEC EDGAR 10-K/10-Q/8-K downloader
        transcripts.py         # Finnhub earnings call transcripts
        market.py              # yfinance price/return data
    signals/
        lazy_prices.py         # TF-IDF cosine similarity (quantitative)
        filing_changes.py      # LLM-enhanced change detection (qualitative)
        regime.py              # 3-state Gaussian HMM regime detector
    agents/
        state.py               # Shared PipelineState definition
        filing_analyst.py      # Filing Analyst (Lazy Prices + LLM)
        earnings_analyst.py    # Earnings Call Analyst (communication scoring)
        news_synthesizer.py    # News Synthesizer (8-K material events)
        coordinator.py         # Research Coordinator (signal combiner)
        pipeline.py            # LangGraph orchestration
    backtest/
        engine.py              # Portfolio construction, performance metrics
        walk_forward.py        # Walk-forward backtest (Lazy Prices, 24mo OOS)
        combined.py            # Combined LLM pipeline + regime filter
        tearsheet.py           # Quantstats HTML report generation
    trading/
        executor.py            # Alpaca paper trading executor
reports/
    tearsheet_*.html           # Generated performance tearsheets
METHODOLOGY.md                 # Full methodology for interview discussion
```

## Results

### Walk-forward (24 months out-of-sample, Lazy Prices signal)

| Variant | Sharpe | Return | Max DD | % Positive |
|---------|--------|--------|--------|------------|
| No filter | -1.49 | -25.3% | -28.1% | 37.5% |
| HMM regime filter | -1.04 | -19.9% | -28.6% | 50.0% |

### Multi-agent LLM pipeline (single-period, 100 tickers)

| Metric | Value |
|--------|-------|
| Net return | +5.64% |
| Short leg | +11.71% (correctly shorted CNC -25%, CPB -18%, BA -17%) |

See [METHODOLOGY.md](METHODOLOGY.md) for full discussion of economic hypothesis, what works, what doesn't, and the three interview questions.

## Tests

```bash
pytest tests/ -v
```
