"""Filing Analyst agent — processes 10-K/10-Q filings for trading signals.

Combines two approaches:
1. Lazy Prices cosine similarity (quantitative)
2. LLM-based change detection (qualitative — risk factors, litigation, guidance)

Implements the Cohen, Malloy & Nguyen (2020) insight: companies that substantially
change their filing language tend to underperform.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from alpha_graph.agents.state import AgentSignal, PipelineState
from alpha_graph.config import cfg

SYSTEM_PROMPT = """\
You are a Filing Analyst at a quantitative trading firm. Your job is to compare
two consecutive SEC filings (10-K or 10-Q) from the same company and assess
whether the changes are bullish, bearish, or neutral for the stock.

Focus on MATERIAL changes in:
- Risk factors: new risks added, risks removed, escalation of existing risks
- Litigation: new lawsuits, settlement changes, regulatory actions
- Revenue/guidance: language shifts around growth, margins, outlook
- MD&A tone: management confidence, uncertainty language, hedging words

Respond with a JSON object:
{
    "signal": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "confidence": <float from 0.0 to 1.0>,
    "rationale": "<2-3 sentence explanation of the most material changes>",
    "key_changes": ["<change 1>", "<change 2>", ...]
}

If filings are substantially similar (no material changes), signal should be
near 0 with moderate confidence. Major new risk factors or litigation should
push the signal strongly negative.
"""

MAX_CHARS = 12_000  # per filing, to fit in context


def _compute_cosine_sim(text_a: str, text_b: str) -> float:
    """Quick TF-IDF cosine similarity between two texts."""
    try:
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix = vec.fit_transform([text_a, text_b])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception:
        return 0.5


def filing_analyst(state: PipelineState) -> dict:
    """Analyze filing changes and produce a trading signal."""
    ticker = state["ticker"]
    filing_text = state.get("filing_text", "")
    prev_filing_text = state.get("prev_filing_text", "")

    if not filing_text or not prev_filing_text:
        logger.info(f"[{ticker}] Filing Analyst: insufficient filing data, skipping")
        return {"signals": []}

    # Quantitative: cosine similarity
    cos_sim = _compute_cosine_sim(prev_filing_text, filing_text)

    # Qualitative: LLM change analysis
    llm = ChatOpenAI(
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        api_key=cfg.openai_api_key,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Ticker: {ticker}\n"
            f"Cosine similarity between filings: {cos_sim:.4f} "
            f"({'high similarity — few changes' if cos_sim > 0.85 else 'notable changes detected'})\n\n"
            f"## Previous Filing\n{prev_filing_text[:MAX_CHARS]}\n\n"
            f"## Current Filing\n{filing_text[:MAX_CHARS]}"
        )),
    ]

    try:
        response = llm.invoke(messages)
        result = json.loads(response.content)
    except Exception as e:
        logger.error(f"[{ticker}] Filing Analyst LLM call failed: {e}")
        # Fallback: use cosine similarity as signal
        # Low similarity = bearish (Lazy Prices)
        fallback_signal = (cos_sim - 0.85) * 5  # maps 0.75-0.95 to roughly -0.5 to +0.5
        fallback_signal = max(-1.0, min(1.0, fallback_signal))
        result = {
            "signal": fallback_signal,
            "confidence": 0.4,
            "rationale": f"LLM unavailable. Cosine similarity: {cos_sim:.4f}",
        }

    signal = AgentSignal(
        ticker=ticker,
        signal=max(-1.0, min(1.0, float(result.get("signal", 0)))),
        confidence=max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
        rationale=result.get("rationale", ""),
        source="filing_analyst",
    )

    logger.info(
        f"[{ticker}] Filing Analyst: signal={signal['signal']:.2f}, "
        f"confidence={signal['confidence']:.2f}, cos_sim={cos_sim:.4f}"
    )

    return {"signals": [signal]}
