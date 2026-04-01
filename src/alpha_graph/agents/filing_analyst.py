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
from alpha_graph.utils.llm import extract_json

SYSTEM_PROMPT = """\
You are a Filing Analyst at a quantitative trading firm. Your job is to compare
two consecutive SEC filings (10-K or 10-Q) from the same company and provide a
BALANCED assessment of both risks AND opportunities for the stock.

## RISK DETECTION (score from -1.0 to 0.0)
Identify NEGATIVE signals:
- Risk factors: new risks added, escalation of existing risks, new litigation
- Regulatory: new compliance issues, government investigations, consent decrees
- Financial stress: rising debt levels, covenant concerns, going concern language
- Operational: supply chain disruptions, key customer losses, margin compression
- Management hedging: increased use of uncertainty language ("may", "could", "uncertain")

## OPPORTUNITY DETECTION (score from 0.0 to +1.0)
Identify POSITIVE signals:
- Revenue growth: new market entry, expanding TAM, accelerating growth language
- Margin expansion: cost reduction programs, operating leverage, economies of scale
- Strategic wins: new material agreements, strategic partnerships, patent grants
- Balance sheet improvement: debt reduction, improved cash flow, share buybacks
- Favorable regulation: regulatory approvals, deregulation benefits, tax advantages
- Competitive moats: increased R&D spending, market share gains, pricing power language
- Risk REMOVAL: previously disclosed risks that have been resolved or reduced

## SCORING INSTRUCTIONS
You MUST score risk and opportunity INDEPENDENTLY:
- risk_score: float from -1.0 (severe new risks) to 0.0 (no new risks)
- opportunity_score: float from 0.0 (no opportunities) to +1.0 (strong catalysts)
- signal = opportunity_score + risk_score (the net effect, balanced)

This ensures both positive and negative signals are properly captured.

Respond with a JSON object:
{
    "risk_score": <float from -1.0 to 0.0>,
    "opportunity_score": <float from 0.0 to 1.0>,
    "signal": <float from -1.0 to 1.0, should equal opportunity_score + risk_score>,
    "confidence": <float from 0.0 to 1.0>,
    "rationale": "<2-3 sentence explanation covering BOTH risks and opportunities>",
    "key_risks": ["<risk 1>", "<risk 2>", ...],
    "key_opportunities": ["<opportunity 1>", "<opportunity 2>", ...]
}

IMPORTANT: Do NOT default to negative signals. Many companies improve year over year.
If filings are substantially similar (no material changes), signal should be near 0.
A company with BOTH new risks AND new opportunities should reflect the balance of both.
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
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url or None,
    )

    # Build context string with similarity interpretation
    sim_interp = "high similarity — few changes" if cos_sim > 0.85 else "notable changes detected"
    if cos_sim < 0.5:
        sim_interp = "major changes detected — filing substantially rewritten"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Ticker: {ticker}\n"
            f"Cosine similarity between filings: {cos_sim:.4f} ({sim_interp})\n\n"
            f"IMPORTANT: Score risks AND opportunities independently.\n"
            f"Look for BOTH warning signs and positive catalysts.\n\n"
            f"## Previous Filing\n{prev_filing_text[:MAX_CHARS]}\n\n"
            f"## Current Filing\n{filing_text[:MAX_CHARS]}"
        )),
    ]

    try:
        response = llm.invoke(messages)
        result = extract_json(response.content)

        # Use the balanced scoring if available
        risk_score = float(result.get("risk_score", 0))
        opportunity_score = float(result.get("opportunity_score", 0))
        if risk_score != 0 or opportunity_score != 0:
            # Use the decomposed scores for a balanced signal
            computed_signal = opportunity_score + risk_score
            result["signal"] = computed_signal
    except Exception as e:
        logger.error(f"[{ticker}] Filing Analyst LLM call failed: {e}")
        # Fallback: use cosine similarity as signal
        # Low similarity = bearish (Lazy Prices), but center at neutral
        fallback_signal = (cos_sim - 0.85) * 5  # maps 0.75-0.95 to roughly -0.5 to +0.5
        fallback_signal = max(-1.0, min(1.0, fallback_signal))
        result = {
            "signal": fallback_signal,
            "confidence": 0.4,
            "rationale": f"LLM unavailable. Cosine similarity: {cos_sim:.4f}",
            "risk_score": min(0, fallback_signal),
            "opportunity_score": max(0, fallback_signal),
        }

    final_signal = max(-1.0, min(1.0, float(result.get("signal", 0))))

    signal = AgentSignal(
        ticker=ticker,
        signal=final_signal,
        confidence=max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
        rationale=result.get("rationale", ""),
        source="filing_analyst",
    )

    logger.info(
        f"[{ticker}] Filing Analyst: signal={signal['signal']:.2f}, "
        f"confidence={signal['confidence']:.2f}, cos_sim={cos_sim:.4f}, "
        f"risk={result.get('risk_score', 'N/A')}, "
        f"opportunity={result.get('opportunity_score', 'N/A')}"
    )

    return {"signals": [signal]}
