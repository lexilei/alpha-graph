"""Synthetic unit tests for C21/C22 — the Lazy Prices edit measures
(signals/lazy_prices_edit.py). Cover the pinned tokenizer, the two identity
cases (identical -> 1, disjoint -> ~0), hand-computable Sim_Simple values, the
replace-both-sides opcode counting, the ratio == Sim_Simple identity, the
chunked fallback path, and C1's pair enumeration / gap filter."""

import difflib

import pytest

from alpha_graph.signals import lazy_prices_edit as lpe


# --------------------------------------------------------------------------- #
# Pinned tokenizer
# --------------------------------------------------------------------------- #

def test_tokenizer_lowercases_and_splits_on_punctuation():
    assert lpe.tokenize("The Quick, Brown FOX!") == ["the", "quick", "brown", "fox"]


def test_tokenizer_keeps_digits_splits_alphanumeric_runs():
    # punctuation (hyphen, %, parens, dot, underscore) delimits; digits kept
    assert lpe.tokenize("Rev up 10-K by 3.5% (item_2)") == [
        "rev", "up", "10", "k", "by", "3", "5", "item", "2"
    ]


def test_tokenizer_case_folds_to_same_token():
    assert lpe.tokenize("AAPL aapl Aapl") == ["aapl", "aapl", "aapl"]


# --------------------------------------------------------------------------- #
# Identity cases
# --------------------------------------------------------------------------- #

def test_identical_documents_similarity_one():
    toks = lpe.tokenize("the company reported strong revenue growth this year")
    m, s, path = lpe.pair_similarity(toks, list(toks))
    assert m == pytest.approx(1.0)
    assert s == pytest.approx(1.0)
    assert path == "full"


def test_disjoint_documents_similarity_zero():
    a = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    b = ["one", "two", "three", "four", "five", "six"]
    m, s, _ = lpe.pair_similarity(a, b)
    # no shared tokens, equal lengths -> exactly 0 on both measures
    assert m == pytest.approx(0.0)
    assert s == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Hand-computable Sim_Simple values
# --------------------------------------------------------------------------- #

def test_known_small_edit_sim_simple_exact():
    # one-token substitution: replace brown->red => adds=1, deletes=1,
    # len_old=len_new=4 => sim_simple = 1 - (1+1)/(4+4) = 0.75
    old = ["the", "quick", "brown", "fox"]
    new = ["the", "quick", "red", "fox"]
    m, s, _ = lpe.pair_similarity(old, new)
    assert s == pytest.approx(0.75)
    assert m == pytest.approx(0.75)


def test_pure_insertion_sim_simple_exact():
    # append two tokens => adds=2, deletes=0, len_old=3 len_new=5
    # sim_simple = 1 - (2+0)/(3+5) = 0.75
    old = ["a", "b", "c"]
    new = ["a", "b", "c", "d", "e"]
    _, s, _ = lpe.pair_similarity(old, new)
    assert s == pytest.approx(0.75)


def test_replace_counts_on_both_sides():
    # replace 1 old token with 2 new: [a,b,c,d] -> [a,x,y,c,d]
    # opcodes: equal[a], replace b->[x,y] (del=1, add=2), equal[c,d]
    old = ["a", "b", "c", "d"]
    new = ["a", "x", "y", "c", "d"]
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    adds, deletes = lpe._adds_deletes(sm)
    assert (adds, deletes) == (2, 1)
    _, s, _ = lpe.pair_similarity(old, new)
    assert s == pytest.approx(1 - 3 / 9)  # 1 - (2+1)/(4+5)


def test_minedit_equals_simple_identity():
    # under the pinned implementations both measures reduce to 2M/T
    old = "the firm faced rising costs and shrinking margins in a tough market".split()
    new = "the firm reported rising costs and growing margins in a strong market".split()
    m, s, _ = lpe.pair_similarity(old, new)
    assert m == pytest.approx(s)


# --------------------------------------------------------------------------- #
# Chunked fallback (forced with small thresholds)
# --------------------------------------------------------------------------- #

def test_chunked_path_selected_above_threshold(monkeypatch):
    monkeypatch.setattr(lpe, "FALLBACK_COMBINED_TOKENS", 6)
    monkeypatch.setattr(lpe, "CHUNK_TOKENS", 3)
    toks = ["a", "b", "c", "d", "e"]  # combined 10 > 6 -> chunked
    m, s, path = lpe.pair_similarity(toks, list(toks))
    assert path == "chunked"
    assert m == pytest.approx(1.0)
    assert s == pytest.approx(1.0)


def test_chunked_identity_and_disjoint(monkeypatch):
    monkeypatch.setattr(lpe, "CHUNK_TOKENS", 4)
    # identical across chunk boundaries -> exactly 1.0
    toks = [str(i % 7) for i in range(30)]
    m, s = lpe._similarity_chunked(toks, list(toks))
    assert m == pytest.approx(1.0)
    assert s == pytest.approx(1.0)
    # aligned chunks fully disjoint -> 0.0, and the identity still holds
    a = [f"x{i}" for i in range(20)]
    b = [f"y{i}" for i in range(20)]
    m2, s2 = lpe._similarity_chunked(a, b)
    assert m2 == pytest.approx(0.0)
    assert s2 == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# C1 pair enumeration reuse (gap filter, filing_date stamping)
# --------------------------------------------------------------------------- #

def test_compute_for_ticker_gap_filter_and_prev_advance(monkeypatch):
    filings = [
        {"filing_date": "2012-02-20", "text": "alpha beta gamma delta epsilon"},
        {"filing_date": "2013-02-18", "text": "alpha beta gamma delta zeta"},   # 363d -> pair
        {"filing_date": "2015-02-20", "text": "alpha beta gamma delta theta"},  # 732d -> skip
        {"filing_date": "2016-02-19", "text": "alpha beta gamma delta iota"},   # 364d -> pair
    ]
    monkeypatch.setattr(lpe, "_load_filing_texts", lambda t, form: filings)
    rows = lpe.compute_for_ticker("XYZ")
    # only the two <=460d consecutive pairs survive; the >460d span is skipped
    assert [r["filing_date"] for r in rows] == ["2013-02-18", "2016-02-19"]
    # score stamped at the NEW filing; prev advances to the immediate
    # predecessor even across the skipped span
    assert rows[1]["prev_filing_date"] == "2015-02-20"
    for r in rows:
        assert 0.0 <= r["sim_minedit_10k"] <= 1.0
        assert r["sim_minedit_10k"] == pytest.approx(r["sim_simple_10k"])


def test_compute_for_ticker_needs_two_filings(monkeypatch):
    monkeypatch.setattr(lpe, "_load_filing_texts",
                        lambda t, form: [{"filing_date": "2020-01-01", "text": "a b c"}])
    assert lpe.compute_for_ticker("XYZ") == []
