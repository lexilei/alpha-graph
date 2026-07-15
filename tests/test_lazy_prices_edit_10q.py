"""Synthetic unit tests for C24 — the 10-Q YoY Lazy Prices min-edit measure and
the freshest-filing combined stream (signals/lazy_prices_edit_10q.py).

Covers: C2's YoY pair enumeration reused verbatim (same-fiscal-quarter ~365d,
NOT adjacent quarters), the pinned digit-free ``[a-z]+`` tokenizer, and the
combined-stream same-day dedup keep-last.
"""

import pandas as pd
import pytest

from alpha_graph.signals import lazy_prices_edit as lpe
from alpha_graph.signals import lazy_prices_edit_10q as m


# --------------------------------------------------------------------------- #
# 8-quarter fixture: two years of quarterly 10-Qs ~91d apart. Each year-2
# quarter's only ~365d-earlier neighbour is the SAME quarter a year before, so
# the correct YoY pairing is unambiguous and distinct from adjacent quarters.
# --------------------------------------------------------------------------- #

_QUARTER_DATES = [
    "2019-02-15", "2019-05-15", "2019-08-15", "2019-11-15",   # year 1: q0..q3
    "2020-02-15", "2020-05-15", "2020-08-15", "2020-11-15",   # year 2: q4..q7
]
_QUARTER_WORDS = ["alpha", "beta", "gamma", "delta"]


def _eight_quarters(y1_fn, y2_fn) -> list[dict]:
    """Build 8 quarterly 10-Q dicts; y1_fn/y2_fn(word) -> text per quarter."""
    out = []
    for k, d in enumerate(_QUARTER_DATES[:4]):
        out.append({"filing_date": d, "text": y1_fn(_QUARTER_WORDS[k])})
    for k, d in enumerate(_QUARTER_DATES[4:]):
        out.append({"filing_date": d, "text": y2_fn(_QUARTER_WORDS[k])})
    return out


# --------------------------------------------------------------------------- #
# YoY pairing respected (C2's enumeration, pinned by construction)
# --------------------------------------------------------------------------- #

def test_yoy_pairing_matches_same_quarter_one_year_earlier(monkeypatch):
    # distinct text per quarter so nothing about the score can mask a mispair
    filings = _eight_quarters(
        lambda w: f"fiscal 2019 report for the {w} quarter of operations",
        lambda w: f"fiscal 2020 report for the {w} quarter of operations",
    )
    monkeypatch.setattr(m, "_load_filing_texts", lambda t, form: filings)
    rows = m.compute_for_ticker("XYZ")

    # exactly the 4 year-2 quarters get a YoY pair; the 4 year-1 quarters have
    # no ~365d-earlier neighbour and produce nothing.
    assert [r["filing_date"] for r in rows] == _QUARTER_DATES[4:]
    # each pairs with the SAME quarter one year earlier — NOT the adjacent
    # prior quarter (that is the whole point of C2's seasonality control).
    assert [r["prev_filing_date"] for r in rows] == _QUARTER_DATES[:4]
    assert rows[0]["prev_filing_date"] == "2019-02-15"      # not 2019-11-15
    for r in rows:
        assert 300 <= r["gap_days"] <= 430
        assert 0.0 <= r["sim_minedit_10q"] <= 1.0


def test_load_filing_texts_is_called_with_10q_form(monkeypatch):
    seen = {}
    def _spy(ticker, form):
        seen["form"] = form
        return []
    monkeypatch.setattr(m, "_load_filing_texts", _spy)
    m.compute_for_ticker("XYZ")
    assert seen["form"] == "10-Q"


def test_needs_four_filings_for_yoy(monkeypatch):
    # three quarters is below C2's >=4 requirement -> no pairs
    filings = [{"filing_date": d, "text": "some report text here"}
               for d in _QUARTER_DATES[:3]]
    monkeypatch.setattr(m, "_load_filing_texts", lambda t, form: filings)
    assert m.compute_for_ticker("XYZ") == []


# --------------------------------------------------------------------------- #
# Pinned digit-free tokenizer ([a-z]+)
# --------------------------------------------------------------------------- #

def test_tokenizer_mode_is_digit_free_alpha():
    assert m.TOKENIZER_MODE == "alpha"


def test_digit_only_changes_score_one_under_digit_free(monkeypatch):
    # year-2 text differs from its year-1 same-quarter pair ONLY in digits
    # (amount 100->250, year 2019->2020): the digit-free tokenizer sees
    # identical token streams -> sim 1.0 on every YoY pair.
    filings = _eight_quarters(
        lambda w: f"net revenue rose to 100 million in fiscal 2019 {w} segment",
        lambda w: f"net revenue rose to 250 million in fiscal 2020 {w} segment",
    )
    monkeypatch.setattr(m, "_load_filing_texts", lambda t, form: filings)
    rows = m.compute_for_ticker("XYZ")
    assert len(rows) == 4
    assert all(r["sim_minedit_10q"] == pytest.approx(1.0) for r in rows)

    # contrast: keeping digits (C21's alnum tokenizer) the same pair scores < 1
    y1 = "net revenue rose to 100 million in fiscal 2019 alpha segment"
    y2 = "net revenue rose to 250 million in fiscal 2020 alpha segment"
    m_alnum, _, _ = lpe.pair_similarity(lpe.tokenize(y1), lpe.tokenize(y2))
    assert m_alnum < 1.0


# --------------------------------------------------------------------------- #
# Combined freshest-filing stream — same-day dedup keep-last (C7's construction)
# --------------------------------------------------------------------------- #

def test_combine_fresh_stream_same_day_keeps_last_10q():
    # ticker X files a 10-K and a 10-Q on the SAME day 2021-03-01; the 10-Q
    # (concatenated second) must win under keep-last.
    k = pd.DataFrame({"ticker": ["X", "X"],
                      "filing_date": ["2020-03-01", "2021-03-01"],
                      "sim_minedit_alpha_10k": [0.50, 0.60]})
    q = pd.DataFrame({"ticker": ["X", "X"],
                      "filing_date": ["2021-03-01", "2021-06-01"],
                      "sim_minedit_10q": [0.90, 0.80]})
    c = m.combine_fresh_stream(k, q)
    assert list(c.columns) == ["ticker", "filing_date", "sim_minedit_fresh"]
    # union is deduped to 3 distinct (ticker, date) rows, sorted by date
    assert c["filing_date"].tolist() == list(pd.to_datetime(
        ["2020-03-01", "2021-03-01", "2021-06-01"]))
    same_day = c.loc[c["filing_date"] == pd.Timestamp("2021-03-01"),
                     "sim_minedit_fresh"].iloc[0]
    assert same_day == pytest.approx(0.90)      # 10-Q kept, not the 0.60 10-K


def test_combine_fresh_stream_pools_both_forms():
    k = pd.DataFrame({"ticker": ["A"], "filing_date": ["2020-12-20"],
                      "sim_minedit_alpha_10k": [0.7]})
    q = pd.DataFrame({"ticker": ["A", "A"],
                      "filing_date": ["2021-03-05", "2021-06-04"],
                      "sim_minedit_10q": [0.8, 0.85]})
    c = m.combine_fresh_stream(k, q)
    assert len(c) == 3                          # no same-day collision -> all kept
    assert set(c["sim_minedit_fresh"]) == {0.7, 0.8, 0.85}
