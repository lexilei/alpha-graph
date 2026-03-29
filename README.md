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

# 3. Run multi-agent pipeline
python -m alpha_graph.agents.pipeline --tickers AAPL MSFT GOOGL

# 4. Backtest
python -m alpha_graph.backtest.engine
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
    agents/
        state.py               # Shared PipelineState definition
        filing_analyst.py      # Filing Analyst (Lazy Prices + LLM)
        earnings_analyst.py    # Earnings Call Analyst (communication scoring)
        news_synthesizer.py    # News Synthesizer (8-K material events)
        coordinator.py         # Research Coordinator (signal combiner)
        pipeline.py            # LangGraph orchestration
    backtest/
        engine.py              # Walk-forward backtest, Deflated Sharpe Ratio
```

## Methodology

- **Walk-forward validation**: purged CV with 5-day gap to prevent leakage
- **Transaction costs**: 10 bps round-trip for liquid large-caps
- **Deflated Sharpe Ratio**: accounts for multiple testing (Bailey & Lopez de Prado 2014)
- **Long-short portfolio**: top/bottom decile by combined signal score

## Tests

```bash
pytest tests/ -v
```
