"""Unit tests for the Research Coordinator signal combiner."""

from alpha_graph.agents.coordinator import research_coordinator
from alpha_graph.agents.state import AgentSignal, PipelineState


def _make_state(signals: list[AgentSignal]) -> PipelineState:
    return PipelineState(
        ticker="TEST",
        company_name="Test Corp",
        filing_text="",
        prev_filing_text="",
        transcript_text="",
        news_8k_text="",
        signals=signals,
        recommendation="",
        combined_score=0.0,
        combined_confidence=0.0,
        combined_rationale="",
    )


def test_coordinator_buy_signal():
    signals = [
        AgentSignal(ticker="TEST", signal=0.7, confidence=0.8, rationale="Strong filing", source="filing_analyst"),
        AgentSignal(ticker="TEST", signal=0.5, confidence=0.7, rationale="Good tone", source="earnings_analyst"),
        AgentSignal(ticker="TEST", signal=0.3, confidence=0.5, rationale="No bad news", source="news_synthesizer"),
    ]
    result = research_coordinator(_make_state(signals))
    assert result["recommendation"] == "BUY"
    assert result["combined_score"] > 0.25


def test_coordinator_sell_signal():
    signals = [
        AgentSignal(ticker="TEST", signal=-0.8, confidence=0.9, rationale="Risk escalation", source="filing_analyst"),
        AgentSignal(ticker="TEST", signal=-0.5, confidence=0.7, rationale="Hedging language", source="earnings_analyst"),
        AgentSignal(ticker="TEST", signal=-0.3, confidence=0.6, rationale="Auditor change", source="news_synthesizer"),
    ]
    result = research_coordinator(_make_state(signals))
    assert result["recommendation"] == "SELL"
    assert result["combined_score"] < -0.25


def test_coordinator_hold_on_disagreement():
    signals = [
        AgentSignal(ticker="TEST", signal=0.8, confidence=0.8, rationale="Bullish filing", source="filing_analyst"),
        AgentSignal(ticker="TEST", signal=-0.7, confidence=0.8, rationale="Bearish tone", source="earnings_analyst"),
    ]
    result = research_coordinator(_make_state(signals))
    # When agents strongly disagree, confidence should be lower
    assert result["combined_confidence"] < 0.6


def test_coordinator_no_signals():
    result = research_coordinator(_make_state([]))
    assert result["recommendation"] == "HOLD"
    assert result["combined_score"] == 0.0
    assert result["combined_confidence"] == 0.0
