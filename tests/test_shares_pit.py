"""Tests for point-in-time shares outstanding (alpha_graph.data.shares_pit).

Synthetic-frame tests pin the PIT semantics: availability is by FILED
date (a fact measured 03-31 but filed 05-10 must not be visible 04-15),
duplicate (end, filed) rows are share classes and sum, and an entity
with no facts yields NaN. The real-cache tests are acceptance gates on
the AAPL 4:1 split (effective 2020-08-31, first disclosed by the 10-K
filed 2020-10-30).
"""

import math

import pandas as pd
import pytest

from alpha_graph.data.shares_pit import (
    SHARES_PIT_PATH,
    market_cap_panel,
    shares_asof,
)

needs_cache = pytest.mark.skipif(
    not SHARES_PIT_PATH.exists(), reason="shares_outstanding_pit.parquet not present")


def _facts(rows):
    df = pd.DataFrame(rows, columns=["end", "filed", "shares"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df


@pytest.fixture()
def facts():
    return _facts([
        ("2024-12-31", "2025-02-01", 90.0),
        ("2025-03-31", "2025-05-10", 100.0),
    ])


# --------------------------------------------------------------------------- #
# PIT semantics on synthetic facts
# --------------------------------------------------------------------------- #

def test_picks_by_filed_not_end(facts):
    # The 2025-03-31 measurement was filed 05-10: invisible on 04-15.
    assert shares_asof(facts, "2025-04-15") == 90.0


def test_visible_from_filed_date_inclusive(facts):
    assert shares_asof(facts, "2025-05-09") == 90.0
    assert shares_asof(facts, "2025-05-10") == 100.0
    assert shares_asof(facts, "2026-01-01") == 100.0


def test_nan_before_first_filing(facts):
    assert math.isnan(shares_asof(facts, "2025-01-15"))


def test_no_facts_is_nan():
    assert math.isnan(shares_asof(_facts([]), "2025-01-15"))


def test_multiclass_same_end_filed_rows_sum():
    two_class = _facts([
        ("2024-12-31", "2025-02-01", 60.0),
        ("2024-12-31", "2025-02-01", 40.0),
    ])
    assert shares_asof(two_class, "2025-03-01") == 100.0


def test_amendment_with_stale_end_cannot_roll_back(facts):
    # A later filing re-reporting the OLD measurement date must not
    # supersede the newer measurement already on file.
    amended = pd.concat(
        [facts, _facts([("2024-12-31", "2025-06-01", 90.0)])], ignore_index=True)
    assert shares_asof(amended, "2025-07-01") == 100.0


def test_same_end_correction_filed_later_wins(facts):
    corrected = pd.concat(
        [facts, _facts([("2025-03-31", "2025-06-01", 101.0)])], ignore_index=True)
    assert shares_asof(corrected, "2025-05-15") == 100.0
    assert shares_asof(corrected, "2025-06-01") == 101.0


def test_vector_dates_return_series(facts):
    dates = ["2025-01-15", "2025-04-15", "2025-05-10"]
    out = shares_asof(facts, dates)
    assert isinstance(out, pd.Series)
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == 90.0
    assert out.iloc[2] == 100.0


def test_multi_ticker_frame_rejected(facts):
    bad = facts.assign(ticker=["A", "B"])
    with pytest.raises(ValueError, match="multiple tickers"):
        shares_asof(bad, "2025-05-15")


def test_market_cap_panel_availability_correct(facts):
    facts_t = facts.assign(ticker="TT")
    prices = pd.DataFrame({
        "ticker": ["TT", "TT", "ZZ"],
        "date": pd.to_datetime(["2025-04-15", "2025-05-15", "2025-05-15"]),
        "close": [2.0, 2.0, 3.0],
    }, index=[7, 3, 11])  # non-default index: alignment must be positional
    caps = market_cap_panel(prices, facts=facts_t)
    assert list(caps.index) == [7, 3, 11]
    assert caps.loc[7] == 180.0      # 90 shares known on 04-15
    assert caps.loc[3] == 200.0      # 100 shares known from 05-10
    assert math.isnan(caps.loc[11])  # ZZ has no facts


# --------------------------------------------------------------------------- #
# Acceptance gates on the real cache (AAPL 4:1 split, effective 2020-08-31)
# --------------------------------------------------------------------------- #

@needs_cache
def test_aapl_post_split_count_visible_from_10k():
    assert shares_asof("AAPL", pd.Timestamp("2020-10-30")) > 15e9


@needs_cache
def test_aapl_pre_split_count_before_disclosure():
    # 2020-07-15 predates the split; the post-split ~17B count (filed
    # 2020-10-30) must not leak back.
    assert shares_asof("AAPL", pd.Timestamp("2020-07-15")) < 5.5e9
