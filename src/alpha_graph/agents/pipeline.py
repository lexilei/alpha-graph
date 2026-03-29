"""Multi-agent LangGraph pipeline — orchestrates all analyst agents.

Architecture:
    START -> fan-out -> [filing_analyst, earnings_analyst, news_synthesizer] (parallel)
          -> fan-in  -> research_coordinator -> END

Each agent reads from shared PipelineState and appends its signal.
The coordinator combines all signals into a final recommendation.

Usage:
    python -m alpha_graph.agents.pipeline [--tickers AAPL MSFT] [--max-tickers 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger

from alpha_graph.agents.coordinator import research_coordinator
from alpha_graph.agents.earnings_analyst import earnings_analyst
from alpha_graph.agents.filing_analyst import filing_analyst
from alpha_graph.agents.news_synthesizer import news_synthesizer
from alpha_graph.agents.state import PipelineState
from alpha_graph.config import CACHE_DIR, FILINGS_DIR, TRANSCRIPTS_DIR


def _fan_out_to_analysts(state: PipelineState) -> list[Send]:
    """Route to available analyst agents based on input data."""
    sends = []
    if state.get("filing_text") and state.get("prev_filing_text"):
        sends.append(Send("filing_analyst", state))
    if state.get("transcript_text"):
        sends.append(Send("earnings_analyst", state))
    # Always run news synthesizer — it handles missing data gracefully
    sends.append(Send("news_synthesizer", state))
    return sends


def build_graph() -> StateGraph:
    """Build and compile the multi-agent pipeline graph."""
    builder = StateGraph(PipelineState)

    # Add agent nodes
    builder.add_node("filing_analyst", filing_analyst)
    builder.add_node("earnings_analyst", earnings_analyst)
    builder.add_node("news_synthesizer", news_synthesizer)
    builder.add_node("research_coordinator", research_coordinator)

    # Fan-out from START to agents in parallel
    builder.add_conditional_edges(START, _fan_out_to_analysts)

    # Fan-in: all agents converge to coordinator
    builder.add_edge("filing_analyst", "research_coordinator")
    builder.add_edge("earnings_analyst", "research_coordinator")
    builder.add_edge("news_synthesizer", "research_coordinator")

    # Coordinator produces final output
    builder.add_edge("research_coordinator", END)

    return builder.compile()


# Module-level compiled graph (reusable across calls)
graph = build_graph()


def _load_latest_filings(ticker: str) -> tuple[str, str, str]:
    """Load the two most recent 10-K filings and latest 8-K for a ticker.

    Returns: (current_filing_text, prev_filing_text, news_8k_text)
    """
    ticker_dir = FILINGS_DIR / ticker
    if not ticker_dir.exists():
        return "", "", ""

    # 10-K filings sorted by date (newest first)
    tenk_files = sorted(ticker_dir.glob("10-K_*.json"), reverse=True)
    current_text, prev_text = "", ""

    for i, f in enumerate(tenk_files[:2]):
        data = json.loads(f.read_text(encoding="utf-8"))
        sections = data.get("sections", {})
        text = "\n\n".join(sections.values())
        if i == 0:
            current_text = text
        else:
            prev_text = text

    # 8-K filings (most recent ones)
    eightk_files = sorted(ticker_dir.glob("8-K_*.json"), reverse=True)
    news_parts = []
    for f in eightk_files[:5]:  # last 5 8-Ks
        data = json.loads(f.read_text(encoding="utf-8"))
        sections = data.get("sections", {})
        text = "\n\n".join(sections.values())
        if text:
            news_parts.append(f"### 8-K Filed {data.get('filing_date', 'unknown')}\n{text}")

    return current_text, prev_text, "\n\n".join(news_parts)


def _load_latest_transcript(ticker: str) -> str:
    """Load the most recent earnings call transcript for a ticker."""
    ticker_dir = TRANSCRIPTS_DIR / ticker
    if not ticker_dir.exists():
        return ""

    transcript_files = sorted(ticker_dir.glob("earnings_*.json"), reverse=True)
    if not transcript_files:
        return ""

    data = json.loads(transcript_files[0].read_text(encoding="utf-8"))
    return data.get("full_text", "")


def analyze_ticker(ticker: str, company_name: str = "") -> dict:
    """Run the full multi-agent pipeline for a single ticker."""
    filing_text, prev_filing_text, news_8k_text = _load_latest_filings(ticker)
    transcript_text = _load_latest_transcript(ticker)

    if not filing_text and not transcript_text:
        logger.warning(f"[{ticker}] No data available — skipping")
        return {}

    input_state: PipelineState = {
        "ticker": ticker,
        "company_name": company_name or ticker,
        "filing_text": filing_text,
        "prev_filing_text": prev_filing_text,
        "transcript_text": transcript_text,
        "news_8k_text": news_8k_text,
        "signals": [],
        "recommendation": "",
        "combined_score": 0.0,
        "combined_confidence": 0.0,
        "combined_rationale": "",
    }

    result = graph.invoke(input_state)
    return result


def analyze_universe(
    tickers: list[str] | None = None,
    max_tickers: int | None = None,
) -> pd.DataFrame:
    """Run the pipeline across multiple tickers and return a signal DataFrame."""
    if tickers is None:
        from alpha_graph.data.universe import get_sp500_tickers

        tickers = get_sp500_tickers(max_tickers=max_tickers)

    logger.info(f"Running multi-agent pipeline for {len(tickers)} tickers...")
    rows = []

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] Analyzing {ticker}...")
        try:
            result = analyze_ticker(ticker)
        except Exception as e:
            logger.error(f"[{ticker}] Pipeline failed: {e}")
            continue

        if not result:
            continue

        rows.append({
            "ticker": ticker,
            "recommendation": result.get("recommendation", "HOLD"),
            "combined_score": result.get("combined_score", 0.0),
            "combined_confidence": result.get("combined_confidence", 0.0),
            "n_signals": len(result.get("signals", [])),
            "rationale": result.get("combined_rationale", ""),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
        out_path = CACHE_DIR / "pipeline_signals.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df)} signals to {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent analysis pipeline")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--max-tickers", type=int, default=10)
    args = parser.parse_args()

    df = analyze_universe(tickers=args.tickers, max_tickers=args.max_tickers)
    if not df.empty:
        print("\n--- Pipeline Results ---")
        print(df[["ticker", "recommendation", "combined_score", "combined_confidence"]].to_string())

        buys = df[df["recommendation"] == "BUY"]
        sells = df[df["recommendation"] == "SELL"]
        print(f"\nBUY: {len(buys)} | HOLD: {len(df) - len(buys) - len(sells)} | SELL: {len(sells)}")


if __name__ == "__main__":
    main()
