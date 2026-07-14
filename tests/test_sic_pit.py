"""Tests for point-in-time SIC classification (alpha_graph.data.sic_pit).

Synthetic-frame tests pin the PIT semantics: the SIC on date D is the
code of the latest 10-K FILED on or before D (boundary inclusive), a
reclassified company returns different codes before/after its change
filing, and an entity with no history yields None. Division mapping is
pinned on the official range table edges plus EDGAR's non-classifiable
9995; anything unmappable is "UNKNOWN". Real-cache tests are acceptance
gates on AAPL (SIC 3571, ELECTRONIC COMPUTERS, stable for decades).
"""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.data.sic_pit import (
    SIC_HISTORY_PATH,
    load_sic_history,
    sic2,
    sic_asof,
    sic_division,
)

needs_cache = pytest.mark.skipif(
    not SIC_HISTORY_PATH.exists(), reason="sic_history.parquet not present")


def _hist(rows):
    df = pd.DataFrame(rows, columns=["ticker", "filing_date", "sic_code"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


@pytest.fixture()
def hist():
    # AAA reclassifies from manufacturing (3571) to software (7372) at its
    # 2017 10-K; BBB is a bank (6021) with a single filing.
    return _hist([
        ("AAA", "2015-02-20", "3571"),
        ("AAA", "2016-02-19", "3571"),
        ("AAA", "2017-02-21", "7372"),
        ("BBB", "2015-03-01", "6021"),
    ])


def _one(hist, ticker):
    return hist[hist["ticker"] == ticker]


# --------------------------------------------------------------------------- #
# as-of semantics on synthetic history
# --------------------------------------------------------------------------- #

def test_picks_latest_filed_on_or_before(hist):
    assert sic_asof(_one(hist, "AAA"), "2016-06-30") == "3571"


def test_boundary_inclusive(hist):
    # A 10-K filed on D is knowable on D; the day before it is not.
    assert sic_asof(_one(hist, "AAA"), "2017-02-21") == "7372"
    assert sic_asof(_one(hist, "AAA"), "2017-02-20") == "3571"


def test_none_before_first_filing(hist):
    assert sic_asof(_one(hist, "AAA"), "2015-02-19") is None


def test_changer_pre_post(hist):
    aaa = _one(hist, "AAA")
    assert sic_asof(aaa, "2016-12-31") == "3571"
    assert sic_asof(aaa, "2020-01-01") == "7372"


def test_vector_dates_align(hist):
    dates = ["2015-01-01", "2015-02-20", "2017-06-30"]
    out = sic_asof(_one(hist, "AAA"), dates)
    assert list(out) == [None, "3571", "7372"]
    assert list(out.index) == list(pd.to_datetime(dates))


def test_empty_history_is_none():
    assert sic_asof(_hist([]), "2020-01-01") is None


def test_multi_ticker_frame_raises(hist):
    with pytest.raises(ValueError, match="multiple tickers"):
        sic_asof(hist, "2020-01-01")


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="lacks columns"):
        sic_asof(pd.DataFrame({"filing_date": []}), "2020-01-01")


# --------------------------------------------------------------------------- #
# division and major-group mapping
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("code,division", [
    ("0100", "Agriculture"),        # division A lower edge
    ("1000", "Mining"),             # division B lower edge
    ("1499", "Mining"),             # division B upper edge
    ("1731", "Construction"),
    ("2834", "Manufacturing"),
    ("3999", "Manufacturing"),      # division D upper edge
    ("4813", "Transport & Utilities"),
    ("5065", "Wholesale"),
    ("5812", "Retail"),
    ("6021", "Finance"),
    ("6798", "Finance"),            # REITs sit inside H
    ("7372", "Services"),           # prepackaged software
    ("8999", "Services"),           # division I upper edge
    ("9721", "Public Admin"),
    ("9995", "Nonclassifiable"),    # EDGAR non-classifiable establishments
])
def test_division_ranges(code, division):
    assert sic_division(code) == division


@pytest.mark.parametrize("code", [None, "", "0000", "abcd", np.nan, "10000"])
def test_missing_codes_are_unknown_everywhere(code):
    assert sic_division(code) == "UNKNOWN"
    assert sic2(code) == "UNKNOWN"


@pytest.mark.parametrize("code,group", [("1850", "18"), ("6900", "69"), ("9800", "98")])
def test_table_gap_codes_have_no_division_but_keep_major_group(code, group):
    # 1800-1999 / 6800-6999 / 9730-9899 are gaps in the official division
    # table; sic2 still extracts the raw major group.
    assert sic_division(code) == "UNKNOWN"
    assert sic2(code) == group


def test_int_and_float_codes_accepted():
    assert sic_division(7372) == "Services"
    assert sic_division(7372.0) == "Services"
    assert sic_division(100) == "Agriculture"  # un-padded agriculture code


def test_sic2_major_group():
    assert sic2("7372") == "73"
    assert sic2("0100") == "01"
    assert sic2(100) == "01"


# --------------------------------------------------------------------------- #
# real-cache acceptance gates
# --------------------------------------------------------------------------- #

@needs_cache
def test_cache_loads_with_usable_codes():
    df = load_sic_history()
    assert {"ticker", "filing_date", "sic_code"} <= set(df.columns)
    assert df["sic_code"].notna().all()
    assert df["sic_code"].str.fullmatch(r"\d{4}").all()


@needs_cache
def test_aapl_is_electronic_computers():
    assert sic_asof("AAPL", "2020-06-30") == "3571"
    assert sic_division(sic_asof("AAPL", "2020-06-30")) == "Manufacturing"


@needs_cache
def test_aapl_none_before_corpus_start():
    assert sic_asof("AAPL", "1994-01-01") is None
