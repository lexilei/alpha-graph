"""Shared state definition for the multi-agent pipeline."""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


class AgentSignal(TypedDict):
    """A single trading signal produced by an analyst agent."""

    ticker: str
    signal: float  # -1.0 (strong sell) to +1.0 (strong buy)
    confidence: float  # 0.0 to 1.0
    rationale: str
    source: str  # which agent produced it


class PipelineState(TypedDict):
    """State shared across the multi-agent pipeline.

    Keys with Annotated reducers accumulate across agents:
    - signals: list append (operator.add) — each agent adds its signal
    - messages: not used for LLM history, just for logging

    Keys without reducers use last-write-wins semantics.
    """

    # --- Input (set once at invocation) ---
    ticker: str
    company_name: str
    filing_text: str  # concatenated Risk Factors + MD&A from latest 10-K
    prev_filing_text: str  # same sections from the previous 10-K
    transcript_text: str  # latest earnings call transcript
    news_8k_text: str  # recent 8-K filings concatenated

    # --- Intermediate (accumulated by agents) ---
    signals: Annotated[list[AgentSignal], operator.add]

    # --- Output (set by coordinator) ---
    recommendation: str  # BUY / SELL / HOLD
    combined_score: float  # -1.0 to +1.0
    combined_confidence: float  # 0.0 to 1.0
    combined_rationale: str
