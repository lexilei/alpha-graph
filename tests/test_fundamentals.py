"""Unit tests for the fundamentals data pipeline."""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.data.fundamentals import (
    FUNDAMENTAL_COLS,
    YF_FIELD_MAP,
    rank_fundamentals,
)


def _make_sample_df() -> pd.DataFrame:
    """Create a sample fundamentals DataFrame."""
    return pd.DataFrame([
        {"ticker": "AAPL", "date": "2025-01-01", "pe_trailing": 30.0,
         "pb_ratio": 40.0, "div_yield": 0.005, "roe": 0.60,
         "gross_margin": 0.45, "debt_equity": 150.0},
        {"ticker": "MSFT", "date": "2025-01-01", "pe_trailing": 35.0,
         "pb_ratio": 12.0, "div_yield": 0.008, "roe": 0.40,
         "gross_margin": 0.70, "debt_equity": 50.0},
        {"ticker": "GOOG", "date": "2025-01-01", "pe_trailing": 25.0,
         "pb_ratio": 6.0, "div_yield": 0.0, "roe": 0.30,
         "gross_margin": 0.55, "debt_equity": 20.0},
        {"ticker": "NVDA", "date": "2025-01-01", "pe_trailing": 60.0,
         "pb_ratio": 50.0, "div_yield": 0.001, "roe": 0.80,
         "gross_margin": 0.75, "debt_equity": 40.0},
    ])


def test_yf_field_map_covers_all_cols():
    assert set(YF_FIELD_MAP.values()) == set(FUNDAMENTAL_COLS)


def test_rank_fundamentals_produces_rank_columns():
    df = _make_sample_df()
    ranked = rank_fundamentals(df)

    for col in FUNDAMENTAL_COLS:
        rank_col = f"{col}_rank"
        assert rank_col in ranked.columns, f"Missing {rank_col}"


def test_rank_values_between_0_and_1():
    df = _make_sample_df()
    ranked = rank_fundamentals(df)

    for col in FUNDAMENTAL_COLS:
        rank_col = f"{col}_rank"
        vals = ranked[rank_col].dropna()
        assert vals.min() >= 0.0
        assert vals.max() <= 1.0


def test_rank_preserves_order():
    df = _make_sample_df()
    ranked = rank_fundamentals(df)

    # NVDA has highest PE (60), should have highest PE rank
    nvda_rank = ranked.loc[ranked["ticker"] == "NVDA", "pe_trailing_rank"].iloc[0]
    goog_rank = ranked.loc[ranked["ticker"] == "GOOG", "pe_trailing_rank"].iloc[0]
    assert nvda_rank > goog_rank


def test_rank_handles_nan():
    df = _make_sample_df()
    df.loc[0, "pe_trailing"] = np.nan  # AAPL PE is NaN

    ranked = rank_fundamentals(df)
    # NaN should remain NaN in rank
    assert pd.isna(ranked.loc[0, "pe_trailing_rank"])
    # Other tickers should still have valid ranks
    assert ranked["pe_trailing_rank"].notna().sum() == 3


def test_rank_handles_all_nan():
    df = _make_sample_df()
    df["pe_trailing"] = np.nan

    ranked = rank_fundamentals(df)
    assert ranked["pe_trailing_rank"].isna().all()


def test_original_columns_preserved():
    df = _make_sample_df()
    ranked = rank_fundamentals(df)

    for col in FUNDAMENTAL_COLS:
        assert col in ranked.columns
    assert "ticker" in ranked.columns
    assert "date" in ranked.columns
