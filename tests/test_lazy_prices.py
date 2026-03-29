"""Unit tests for the Lazy Prices cosine similarity signal."""

import pandas as pd
import pytest

from alpha_graph.signals.lazy_prices import generate_signal


def _make_df(similarities: list[float]) -> pd.DataFrame:
    """Helper to create a test DataFrame."""
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(len(similarities))],
        "form_type": "10-K",
        "filing_date": pd.date_range("2023-03-01", periods=len(similarities), freq="7D"),
        "prev_filing_date": pd.date_range("2022-03-01", periods=len(similarities), freq="7D"),
        "cosine_similarity": similarities,
        "text_length_curr": 10000,
        "text_length_prev": 10000,
        "text_length_change_pct": 0.0,
    })


def test_generate_signal_assigns_quintiles():
    # 10 stocks: first 2 should be short (-1), last 2 should be long (1), middle neutral (0)
    sims = [0.1, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.9, 0.95]
    df = _make_df(sims)
    result = generate_signal(df)

    assert "signal" in result.columns
    assert result["signal"].iloc[0] == -1  # lowest similarity -> short
    assert result["signal"].iloc[-1] == 1  # highest similarity -> long


def test_generate_signal_empty():
    df = pd.DataFrame()
    result = generate_signal(df)
    assert result.empty


def test_generate_signal_preserves_columns():
    sims = [0.3, 0.5, 0.7, 0.9, 0.95]
    df = _make_df(sims)
    result = generate_signal(df)

    expected_cols = {"ticker", "cosine_similarity", "signal", "filing_date", "filing_year"}
    assert expected_cols.issubset(set(result.columns))
