"""News Synthesizer agent — monitors 8-K filings for material events.

8-K filings report unscheduled material events: leadership changes, M&A,
asset sales, restatements, delisting notices, etc. These are among the
highest-signal SEC filings because they report *events*, not periodic updates.

Key 8-K items and their typical market impact:
- Item 1.01: Entry into a material agreement (neutral to positive)
- Item 1.02: Termination of a material agreement (negative)
- Item 2.01: Completion of acquisition/disposition (depends on terms)
- Item 2.02: Results of operations (pre-earnings leak, high signal)
- Item 2.05: Costs associated with exit/restructuring (negative short-term)
- Item 2.06: Material impairments (negative)
- Item 4.01: Change in auditor (red flag)
- Item 5.02: Director/officer departure (depends on context)
- Item 8.01: Other events (catch-all, requires analysis)
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from alpha_graph.agents.state import AgentSignal, PipelineState
from alpha_graph.config import cfg
from alpha_graph.utils.llm import extract_json

SYSTEM_PROMPT = """\
You are a News Synthesizer at a quantitative trading firm. Your job is to
analyze recent 8-K filings and material news events to produce a trading signal.

For each 8-K filing, assess:
1. **Event type**: What category of material event is this?
2. **Market impact**: How should this affect the stock price?
3. **Urgency**: Is this a slow-burn issue or requires immediate action?

Key red flags (bearish):
- Change in auditor (Item 4.01)
- Material impairment charges (Item 2.06)
- Termination of material agreements (Item 1.02)
- Officer departures with no immediate replacement (Item 5.02)
- Restatements or non-reliance on prior financials (Item 4.02)

Key positive signals (bullish):
- New material agreements with major customers/partners (Item 1.01)
- Accretive acquisitions at reasonable valuations (Item 2.01)
- Completion of share buyback programs (Item 8.01)

Respond with a JSON object:
{
    "signal": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "confidence": <float from 0.0 to 1.0>,
    "rationale": "<2-3 sentence summary>",
    "events": [
        {
            "item": "<8-K item number>",
            "description": "<what happened>",
            "impact": "<bearish/neutral/bullish>"
        }
    ]
}

If no recent 8-K filings exist, return signal=0 with confidence=0.1 (no news is mildly positive).
"""

MAX_NEWS_CHARS = 15_000


def news_synthesizer(state: PipelineState) -> dict:
    """Analyze 8-K filings and news events to produce a trading signal."""
    ticker = state["ticker"]
    news_text = state.get("news_8k_text", "")

    if not news_text or len(news_text) < 50:
        logger.info(f"[{ticker}] News Synthesizer: no 8-K data, returning neutral")
        # No news is mildly positive (absence of negative events)
        signal = AgentSignal(
            ticker=ticker,
            signal=0.05,
            confidence=0.1,
            rationale="No recent 8-K filings — no material events detected.",
            source="news_synthesizer",
        )
        return {"signals": [signal]}

    llm = ChatOpenAI(
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url or None,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Ticker: {ticker}\n"
            f"Company: {state.get('company_name', ticker)}\n\n"
            f"## Recent 8-K Filings\n{news_text[:MAX_NEWS_CHARS]}"
        )),
    ]

    try:
        response = llm.invoke(messages)
        result = extract_json(response.content)
    except Exception as e:
        logger.error(f"[{ticker}] News Synthesizer LLM call failed: {e}")
        return {"signals": []}

    signal = AgentSignal(
        ticker=ticker,
        signal=max(-1.0, min(1.0, float(result.get("signal", 0)))),
        confidence=max(0.0, min(1.0, float(result.get("confidence", 0.3)))),
        rationale=result.get("rationale", ""),
        source="news_synthesizer",
    )

    events = result.get("events", [])
    logger.info(
        f"[{ticker}] News Synthesizer: signal={signal['signal']:.2f}, "
        f"confidence={signal['confidence']:.2f}, events={len(events)}"
    )

    return {"signals": [signal]}
