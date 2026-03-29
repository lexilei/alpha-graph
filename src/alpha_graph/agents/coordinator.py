"""Research Coordinator — combines signals from all analyst agents.

This is the final node in the pipeline. It takes the accumulated signals
from the Filing Analyst, Earnings Call Analyst, and News Synthesizer,
and produces a single combined recommendation with confidence-weighted scoring.
"""

from __future__ import annotations

from loguru import logger

from alpha_graph.agents.state import PipelineState


# Weight each agent's signal by source reliability
AGENT_WEIGHTS = {
    "filing_analyst": 0.40,  # highest weight — Lazy Prices is well-documented
    "earnings_analyst": 0.35,  # strong academic backing
    "news_synthesizer": 0.25,  # event-driven, high impact but sporadic
}


def research_coordinator(state: PipelineState) -> dict:
    """Combine agent signals into a final recommendation.

    Uses confidence-weighted averaging:
        combined = sum(signal_i * confidence_i * weight_i) / sum(confidence_i * weight_i)
    """
    ticker = state["ticker"]
    signals = state.get("signals", [])

    if not signals:
        logger.warning(f"[{ticker}] Coordinator: no signals received")
        return {
            "recommendation": "HOLD",
            "combined_score": 0.0,
            "combined_confidence": 0.0,
            "combined_rationale": "No analyst signals available.",
        }

    # Confidence-weighted combination
    weighted_sum = 0.0
    weight_total = 0.0
    rationale_parts = []

    for sig in signals:
        source = sig["source"]
        w = AGENT_WEIGHTS.get(source, 0.2)
        weighted_sum += sig["signal"] * sig["confidence"] * w
        weight_total += sig["confidence"] * w
        rationale_parts.append(
            f"[{source}] signal={sig['signal']:+.2f} conf={sig['confidence']:.2f}: "
            f"{sig['rationale']}"
        )

    combined_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    combined_score = max(-1.0, min(1.0, combined_score))

    # Combined confidence: weighted average of individual confidences,
    # boosted if agents agree, penalized if they disagree
    avg_confidence = sum(s["confidence"] for s in signals) / len(signals)
    signal_values = [s["signal"] for s in signals]
    agreement = 1.0 - (max(signal_values) - min(signal_values)) / 2.0 if len(signal_values) > 1 else 0.5
    combined_confidence = avg_confidence * (0.5 + 0.5 * agreement)
    combined_confidence = max(0.0, min(1.0, combined_confidence))

    # Map score to recommendation
    if combined_score > 0.25 and combined_confidence > 0.3:
        recommendation = "BUY"
    elif combined_score < -0.25 and combined_confidence > 0.3:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    combined_rationale = (
        f"{recommendation} (score={combined_score:+.3f}, confidence={combined_confidence:.3f})\n"
        + "\n".join(rationale_parts)
    )

    logger.info(
        f"[{ticker}] Coordinator: {recommendation} "
        f"score={combined_score:+.3f} confidence={combined_confidence:.3f} "
        f"({len(signals)} signals)"
    )

    return {
        "recommendation": recommendation,
        "combined_score": combined_score,
        "combined_confidence": combined_confidence,
        "combined_rationale": combined_rationale,
    }
