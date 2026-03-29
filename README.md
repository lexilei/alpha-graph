# alpha-graph

Financial knowledge graph and multi-agent LLM pipeline for quantitative trading signals.

## Strategy overview

1. **Multi-agent LLM pipeline** — Specialized agents process SEC filings, earnings calls, and news to generate trading signals based on the "Lazy Prices" anomaly (Cohen, Malloy & Nguyen 2020) and executive communication patterns.

2. **Financial knowledge graph** — Extract supplier/customer/competitor relationships from 10-K filings, build a dynamic knowledge graph, and predict return spillovers via GNN.

3. **Regime-aware pairs trading** — HMM regime detection overlay with Hierarchical Risk Parity portfolio construction.

## Setup

```bash
# Clone and install
git clone https://github.com/lexilei/alpha-graph.git
cd alpha-graph
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your credentials
```

### Required API keys

| Service | Purpose | Cost |
|---------|---------|------|
| SEC EDGAR | 10-K/10-Q filings | Free (just need name + email) |
| Finnhub | Earnings transcripts, news | Free tier (60 req/min) |
| OpenAI | GPT-4o-mini for filing analysis | ~$20-40/month |
| Alpaca | Market data + paper trading | Free |

## Usage

```bash
# Download SEC filings for S&P 500 companies
python -m alpha_graph.data.filings

# Compute Lazy Prices similarity scores
python -m alpha_graph.signals.lazy_prices

# Fetch earnings call transcripts
python -m alpha_graph.data.transcripts
```

## Project structure

```
src/alpha_graph/
    config.py          # Central configuration
    data/
        universe.py    # S&P 500 ticker management
        filings.py     # SEC EDGAR filing downloader
        transcripts.py # Earnings call transcript collector
    signals/
        lazy_prices.py # Cosine similarity between consecutive filings
    agents/            # LLM agent pipeline (Week 3-4)
    utils/
```
