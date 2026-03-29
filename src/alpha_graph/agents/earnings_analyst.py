"""Earnings Call Analyst agent — scores executive communication patterns.

Based on research showing "proactive and on-topic" executives generate 247 bps
of alpha. Analyzes:
- Tone and sentiment (confidence vs hedging language)
- Specificity (concrete numbers vs vague qualifiers)
- Q&A dynamics (deflection, non-answers, analyst pushback)
- Forward guidance language
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from alpha_graph.agents.state import AgentSignal, PipelineState
from alpha_graph.config import cfg

SYSTEM_PROMPT = """\
You are an Earnings Call Analyst at a quantitative trading firm. Your job is to
analyze earnings call transcripts to assess executive communication quality and
extract a trading signal.

Evaluate:
1. **Management tone** (0-10): Confident and specific vs vague and hedging?
   - Bearish cues: "challenging environment", "headwinds", excessive use of
     "approximately", pivoting away from questions
   - Bullish cues: specific numbers, raised guidance, confident language about
     pipeline/backlog, direct answers to tough questions

2. **Guidance quality** (0-10): Is forward guidance raised, maintained, or lowered?
   Are they sandbagging or being aggressive?

3. **Q&A dynamics** (0-10): How do executives handle analyst questions?
   - Bearish: deflecting, non-answers, CFO jumping in to "clarify" CEO comments
   - Bullish: detailed answers, volunteering additional context, addressing risks head-on

4. **Proactiveness** (0-10): Does management proactively address risks and concerns,
   or do they bury bad news and hope analysts don't ask?

Respond with a JSON object:
{
    "signal": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "confidence": <float from 0.0 to 1.0>,
    "rationale": "<2-3 sentence summary of your assessment>",
    "scores": {
        "management_tone": <0-10>,
        "guidance_quality": <0-10>,
        "qa_dynamics": <0-10>,
        "proactiveness": <0-10>
    }
}
"""

MAX_TRANSCRIPT_CHARS = 20_000


def earnings_analyst(state: PipelineState) -> dict:
    """Analyze earnings call transcript and produce a trading signal."""
    ticker = state["ticker"]
    transcript = state.get("transcript_text", "")

    if not transcript or len(transcript) < 200:
        logger.info(f"[{ticker}] Earnings Analyst: no transcript available, skipping")
        return {"signals": []}

    llm = ChatOpenAI(
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        api_key=cfg.openai_api_key,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Ticker: {ticker}\n"
            f"Company: {state.get('company_name', ticker)}\n\n"
            f"## Earnings Call Transcript\n{transcript[:MAX_TRANSCRIPT_CHARS]}"
        )),
    ]

    try:
        response = llm.invoke(messages)
        result = json.loads(response.content)
    except Exception as e:
        logger.error(f"[{ticker}] Earnings Analyst LLM call failed: {e}")
        return {"signals": []}

    signal = AgentSignal(
        ticker=ticker,
        signal=max(-1.0, min(1.0, float(result.get("signal", 0)))),
        confidence=max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
        rationale=result.get("rationale", ""),
        source="earnings_analyst",
    )

    scores = result.get("scores", {})
    logger.info(
        f"[{ticker}] Earnings Analyst: signal={signal['signal']:.2f}, "
        f"confidence={signal['confidence']:.2f}, "
        f"tone={scores.get('management_tone', '?')}, "
        f"guidance={scores.get('guidance_quality', '?')}"
    )

    return {"signals": [signal]}
