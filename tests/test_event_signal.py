"""Unit tests for the 8-K event-driven signal module."""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.signals.event_signal import (
    ITEM_SCORES,
    _extract_item_numbers,
    score_filing,
    compute_event_signal,
)


def test_extract_item_numbers_basic():
    text = "Item 1.01 - Entry into a Material Agreement"
    items = _extract_item_numbers(text)
    assert "1.01" in items


def test_extract_item_numbers_multiple():
    text = (
        "Item 2.05 - Costs Associated with Exit Activities\n"
        "Item 9.01 - Financial Statements and Exhibits\n"
        "Item 5.02 - Departure of Directors"
    )
    items = _extract_item_numbers(text)
    assert "2.05" in items
    assert "9.01" in items
    assert "5.02" in items
    assert len(items) == 3


def test_extract_item_numbers_case_insensitive():
    text = "ITEM 4.01 - Changes in Auditor"
    items = _extract_item_numbers(text)
    assert "4.01" in items


def test_extract_item_numbers_empty():
    items = _extract_item_numbers("No items here")
    assert items == []


def test_score_filing_auditor_change():
    """Auditor change is the most negative single-event signal."""
    score = score_filing(["4.01"])
    assert score == -0.7


def test_score_filing_material_agreement():
    score = score_filing(["1.01"])
    assert score == 0.3


def test_score_filing_multiple_items_picks_extreme():
    """When multiple items, return the most extreme score."""
    score = score_filing(["1.01", "4.01"])  # +0.3 and -0.7
    # -0.7 has larger absolute value, so it should be returned
    assert score == -0.7


def test_score_filing_neutral_items():
    score = score_filing(["7.01", "8.01", "9.01"])
    assert score == 0.0


def test_score_filing_empty():
    score = score_filing([])
    assert score == 0.0


def test_item_scores_completeness():
    """Ensure all common 8-K items have a defined score."""
    common_items = ["1.01", "1.02", "2.01", "2.05", "4.01", "5.02", "7.01", "8.01"]
    for item in common_items:
        assert item in ITEM_SCORES, f"Missing score for Item {item}"


def test_item_scores_range():
    """All scores should be in [-1, +1]."""
    for item, score in ITEM_SCORES.items():
        assert -1.0 <= score <= 1.0, f"Item {item} score {score} out of range"
