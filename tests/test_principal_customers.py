"""Synthetic tests for principal-customer extraction (no LLM, no network)."""

import pandas as pd
import pytest

from alpha_graph.data.principal_customers import (
    CONCENTRATION_RE,
    _coerce_pct,
    _norm_name,
    _validate_customers,
    build_alias_map,
    compute_validity,
    finalize_edges,
    resolve_customer,
    select_for_extraction,
)

TEXT = (
    "During fiscal 2023, Wal-Mart Stores, Inc. accounted for approximately "
    "26% of our consolidated net sales. Sales to the U.S. Government were "
    "12% of revenues. No other customer accounted for 10% or more of our "
    "net sales."
)


# ---------------------------------------------------------------------------
# Verbatim enforcement
# ---------------------------------------------------------------------------

def test_verbatim_enforcement_drops_paraphrase():
    raw = [
        {   # exact quote -> kept
            "customer_name": "Wal-Mart Stores, Inc.",
            "revenue_pct": 26,
            "evidence": ("Wal-Mart Stores, Inc. accounted for approximately "
                         "26% of our consolidated net sales"),
        },
        {   # paraphrased evidence -> rejected
            "customer_name": "U.S. Government",
            "revenue_pct": 12,
            "evidence": "The US Government represented about 12% of total revenue",
        },
    ]
    rows, n_evid, n_name = _validate_customers(raw, TEXT)
    assert len(rows) == 2  # both retained in cache rows, with flags
    kept = [r for r in rows if r["evidence_verified"] and r["name_verified"]]
    assert len(kept) == 1
    assert kept[0]["customer_name"] == "Wal-Mart Stores, Inc."
    assert kept[0]["revenue_pct"] == 26.0
    assert n_evid == 1 and n_name == 0


def test_verbatim_tolerates_whitespace_and_case_only():
    # Whitespace runs and case differences are normalized away (same
    # normalization as relationships._evidence_in_text) ...
    raw = [{
        "customer_name": "wal-mart stores,  inc.",
        "revenue_pct": None,
        "evidence": "wal-mart stores, inc.  accounted for\napproximately 26% of",
    }]
    rows, n_evid, _ = _validate_customers(raw, TEXT)
    assert n_evid == 0 and rows[0]["evidence_verified"]
    # ... but any rewording fails, even a single dropped word.
    raw2 = [{
        "customer_name": "Wal-Mart Stores, Inc.",
        "revenue_pct": 26,
        "evidence": "Wal-Mart Stores, Inc. accounted for 26% of our consolidated net sales",
    }]
    rows2, n_evid2, _ = _validate_customers(raw2, TEXT)
    assert n_evid2 == 1 and not rows2[0]["evidence_verified"]


def test_unnamed_or_empty_rows_are_ignored():
    rows, _, _ = _validate_customers(
        [{"customer_name": "", "evidence": "x"}, {"revenue_pct": 5}, "junk"],
        TEXT,
    )
    assert rows == []


def test_coerce_pct():
    assert _coerce_pct(26) == 26.0
    assert _coerce_pct("approximately 12.5%") == 12.5
    assert _coerce_pct(None) is None
    assert _coerce_pct("no percentage") is None
    assert _coerce_pct(250) is None
    assert _coerce_pct(0) is None


# ---------------------------------------------------------------------------
# Entity normalization / alias resolution
# ---------------------------------------------------------------------------

def test_entity_alias_resolution():
    names_by_ticker = {
        "WMT": ["Walmart Inc.", "Wal-Mart Stores, Inc."],
        "T": ["AT&T Inc."],
        "HPQ": ["HP Inc.", "Hewlett-Packard Company"],
        "AAPL": ["Apple Inc."],
        "GOOGL": ["Alphabet Inc."],
        "BA": ["The Boeing Company"],
    }
    amap = build_alias_map(names_by_ticker)
    # alias case 1: Walmart spelling variants
    assert resolve_customer("Wal-Mart Stores, Inc.", amap) == "WMT"
    assert resolve_customer("Walmart", amap) == "WMT"
    # alias case 2: AT&T with suffix variants
    assert resolve_customer("AT&T Corp.", amap) == "T"
    assert resolve_customer("AT&T", amap) == "T"
    # alias case 3: Hewlett-Packard / HP
    assert resolve_customer("Hewlett-Packard Company", amap) == "HPQ"
    assert resolve_customer("HP", amap) == "HPQ"
    # historical name via hand alias
    assert resolve_customer("Apple Computer, Inc.", amap) == "AAPL"
    assert resolve_customer("Google Inc.", amap) == "GOOGL"
    # "The X Company" normalization from panel names alone
    assert resolve_customer("Boeing", amap) == "BA"
    # non-panel names resolve to None
    assert resolve_customer("Siemens AG", amap) is None
    assert resolve_customer("the U.S. Government", amap) is None


def test_norm_name_suffix_and_punctuation():
    assert _norm_name("The Boeing Company") == "BOEING"
    assert _norm_name("Wal-Mart Stores, Inc.") == "WAL MART STORES"
    assert _norm_name("Deere & Company") == "DEERE"
    assert _norm_name("Amazon.com, Inc.") == "AMAZONCOM"
    assert _norm_name("AT&T Inc.") == "AT&T"
    # affiliates tails, parenthetical aliases, SEC state markers
    assert (_norm_name("Wal-Mart Stores, Inc. and its affiliates (Wal-Mart)")
            == "WAL MART STORES")
    assert _norm_name("Walmart Inc. and its affiliates") == "WALMART"
    assert _norm_name("Tech Data Corporation and its global affiliates") \
        == "TECH DATA"
    assert _norm_name("OCCIDENTAL PETROLEUM CORP /DE/") \
        == "OCCIDENTAL PETROLEUM"
    # U.S. government variants collapse to one canonical key
    assert _norm_name("the U.S. Government") == "US GOVERNMENT"
    assert _norm_name("the U.S. federal government") == "US GOVERNMENT"
    assert _norm_name("the United States Government") == "US GOVERNMENT"
    # but foreign/state governments do not
    assert _norm_name("the Canadian government") != "US GOVERNMENT"


def test_affiliate_tail_resolves_to_panel_ticker():
    amap = build_alias_map({"WMT": ["Walmart Inc."],
                            "OXY": ["OCCIDENTAL PETROLEUM CORP /DE/"]})
    assert resolve_customer(
        "Wal-Mart Stores, Inc. and its affiliates (Wal-Mart)", amap) == "WMT"
    assert resolve_customer("Occidental Petroleum Corporation", amap) == "OXY"


def test_hand_alias_only_applies_to_panel_tickers():
    amap = build_alias_map({"AAPL": ["Apple Inc."]})
    assert resolve_customer("Walmart", amap) is None  # WMT not in panel here


# ---------------------------------------------------------------------------
# Validity windows
# ---------------------------------------------------------------------------

def _d(s):
    return pd.Timestamp(s)


def test_validity_windows_redisclosed_extends_and_absent_closes():
    edges = pd.DataFrame([
        {"supplier_ticker": "X", "filing_date": _d("2011-03-01"),
         "customer_key": "WMT"},
        {"supplier_ticker": "X", "filing_date": _d("2012-03-01"),
         "customer_key": "WMT"},
        {"supplier_ticker": "X", "filing_date": _d("2011-03-01"),
         "customer_key": "n:ACME"},
        {"supplier_ticker": "X", "filing_date": _d("2013-03-01"),
         "customer_key": "TGT"},
    ])
    inventory = pd.DataFrame([
        {"supplier_ticker": "X", "filing_date": _d("2011-03-01"),
         "known": True, "keys": ("WMT", "n:ACME")},
        {"supplier_ticker": "X", "filing_date": _d("2012-03-01"),
         "known": True, "keys": ("WMT",)},
        {"supplier_ticker": "X", "filing_date": _d("2013-03-01"),
         "known": True, "keys": ("TGT",)},
    ])
    out = compute_validity(edges, inventory)
    # WMT edge from 2011: re-disclosed in 2012 -> window extends past it,
    # closes at the 2013 filing (which drops WMT).
    assert out.loc[0, "valid_to"] == _d("2013-03-01")
    assert out.loc[1, "valid_to"] == _d("2013-03-01")
    # ACME absent from the 2012 filing -> closes there.
    assert out.loc[2, "valid_to"] == _d("2012-03-01")
    # TGT first disclosed in the latest filing -> still open.
    assert pd.isna(out.loc[3, "valid_to"])
    assert (out["valid_from"] == out["filing_date"]).all()


def test_validity_windows_unmatched_closes_unknown_skipped():
    edges = pd.DataFrame([
        {"supplier_ticker": "Y", "filing_date": _d("2011-06-01"),
         "customer_key": "WMT"},
    ])
    # 2012 filing unprocessed-but-matched (unknown) -> skipped;
    # 2013 filing failed the prefilter (known non-disclosure) -> closes.
    inventory = pd.DataFrame([
        {"supplier_ticker": "Y", "filing_date": _d("2011-06-01"),
         "known": True, "keys": ("WMT",)},
        {"supplier_ticker": "Y", "filing_date": _d("2012-06-01"),
         "known": False, "keys": ()},
        {"supplier_ticker": "Y", "filing_date": _d("2013-06-01"),
         "known": True, "keys": ()},
    ])
    out = compute_validity(edges, inventory)
    assert out.loc[0, "valid_to"] == _d("2013-06-01")


# ---------------------------------------------------------------------------
# Non-panel customers kept but flagged; dedup
# ---------------------------------------------------------------------------

def test_nonpanel_customer_kept_but_flagged():
    amap = build_alias_map({"WMT": ["Walmart Inc."], "SUP": ["Supplier Co."]})
    edges = pd.DataFrame([
        {"supplier_ticker": "SUP", "accession": "a1",
         "filing_date": _d("2020-02-01"), "customer_name": "Walmart Inc.",
         "revenue_pct": 20.0, "evidence": "e1", "model": "m"},
        {"supplier_ticker": "SUP", "accession": "a1",
         "filing_date": _d("2020-02-01"), "customer_name": "the U.S. Government",
         "revenue_pct": 15.0, "evidence": "e2", "model": "m"},
        {"supplier_ticker": "SUP", "accession": "a1",
         "filing_date": _d("2020-02-01"), "customer_name": "Airbus S.A.S.",
         "revenue_pct": None, "evidence": "e3", "model": "m"},
    ])
    inventory = pd.DataFrame([
        {"supplier_ticker": "SUP", "filing_date": _d("2020-02-01"),
         "known": True, "keys": ("WMT", "n:US GOVERNMENT", "n:AIRBUS SAS")},
    ])
    out = finalize_edges(edges, amap, inventory)
    # both non-panel rows KEPT — distinct non-panel customers in the same
    # filing must not collapse into one via a shared missing key
    assert len(out) == 3
    gov = out[out["customer_name_raw"] == "the U.S. Government"].iloc[0]
    assert pd.isna(gov["customer_ticker"])
    assert not gov["tradeable"]
    # the key must be the canonical name key, never NaN (float-NaN tickers
    # are truthy — regression guard)
    assert gov["customer_key"] == "n:US GOVERNMENT"
    wmt = out[out["customer_name_raw"] == "Walmart Inc."].iloc[0]
    assert wmt["customer_ticker"] == "WMT"
    assert wmt["tradeable"]


def test_finalize_dedupes_same_customer_within_filing():
    amap = build_alias_map({"WMT": ["Walmart Inc."], "SUP": ["Supplier Co."]})
    edges = pd.DataFrame([
        {"supplier_ticker": "SUP", "accession": "a1",
         "filing_date": _d("2020-02-01"), "customer_name": "Walmart",
         "revenue_pct": None, "evidence": "short", "model": "m"},
        {"supplier_ticker": "SUP", "accession": "a1",
         "filing_date": _d("2020-02-01"), "customer_name": "Walmart Inc.",
         "revenue_pct": 22.0, "evidence": "with pct", "model": "m"},
    ])
    inventory = pd.DataFrame([
        {"supplier_ticker": "SUP", "filing_date": _d("2020-02-01"),
         "known": True, "keys": ("WMT",)},
    ])
    out = finalize_edges(edges, amap, inventory)
    assert len(out) == 1
    assert out.iloc[0]["revenue_pct"] == 22.0  # pct-bearing row wins


# ---------------------------------------------------------------------------
# Prefilter regex + cost gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snippet", [
    "Our principal customers include large retailers.",
    "one major customer accounted for a significant portion",
    "Wal-Mart, our largest customer,",
    "accounted for approximately 12.3% of net sales",
    "No customer accounted for 10% or more of our consolidated net revenues",
    "Concentration of Credit Risk",
    # recall extensions: SWKS-style phrasing
    "Customer Concentration",
    "Apple was approximately 66%, 65% and 63% of our net revenue",
    "were 82% and 80% of aggregate gross accounts receivable",
    "customers accounting for ten percent or more of revenues",
])
def test_prefilter_patterns_match(snippet):
    assert CONCENTRATION_RE.search(snippet)


def test_prefilter_patterns_do_not_match_plain_text():
    assert not CONCENTRATION_RE.search(
        "We sell products through distributors and retailers worldwide.")


def test_cost_gate_under_budget_takes_matched_only():
    pre = pd.DataFrame({
        "supplier_ticker": ["A", "A", "B", "C"],
        "filing_date": pd.to_datetime(
            ["2020-01-01", "2021-01-01", "2020-06-01", "2021-06-01"]),
        "matched": [True, False, True, False],
        "path": ["pa1", "pa2", "pb1", "pc1"],
        "accession": ["1", "2", "3", "4"],
    })
    sel, note = select_for_extraction(pre, budget=10)
    assert len(sel) == 2 and sel["matched"].all()
    assert "all matched" in note


def test_cost_gate_over_budget_expands_ever_matching_tickers():
    pre = pd.DataFrame({
        "supplier_ticker": ["A", "A", "B", "C", "D"],
        "filing_date": pd.to_datetime(
            ["2020-01-01", "2021-01-01", "2020-06-01",
             "2021-06-01", "2022-06-01"]),
        "matched": [True, False, True, False, False],
        "path": list("abcde"),
        "accession": list("12345"),
    })
    sel, note = select_for_extraction(pre, budget=1)
    # all filings of ever-matching tickers A, B (3) + 1 most-recent other (D)
    assert set(sel["supplier_ticker"]) == {"A", "B", "D"}
    assert len(sel) == 4
    assert "ever-matching" in note
