"""Unit tests for relationship extraction guards (confidence + evidence)."""

import json

import pandas as pd
import pytest

from alpha_graph.data.relationships import (
    _evidence_in_text,
    _parse_confidence,
    _validate_relationships,
    audit_evidence,
)


# --- confidence guard ---

def test_parse_confidence_valid_values():
    assert _parse_confidence(0.9) == 0.9
    assert _parse_confidence("0.7") == 0.7
    assert _parse_confidence(1) == 1.0


def test_parse_confidence_clamped():
    assert _parse_confidence(1.7) == 1.0
    assert _parse_confidence(-0.2) == 0.0


def test_parse_confidence_invalid_returns_none():
    assert _parse_confidence(None) is None
    assert _parse_confidence(float("nan")) is None
    assert _parse_confidence(float("inf")) is None
    assert _parse_confidence("high") is None


def test_validate_relationships_drops_and_counts_bad_confidence():
    """Missing/None/NaN confidence rows are dropped explicitly and counted."""
    universe = {"AAPL", "MSFT", "NVDA", "AMD"}
    text = "We compete with Microsoft and buy chips from NVIDIA."
    raw = [
        {"target_ticker": "MSFT", "relation": "competitor", "confidence": 0.9,
         "evidence": "compete with Microsoft"},
        {"target_ticker": "NVDA", "relation": "supplier",
         "confidence": float("nan"), "evidence": "buy chips from NVIDIA"},
        {"target_ticker": "AMD", "relation": "competitor", "confidence": None,
         "evidence": ""},
        {"target_ticker": "AAPL", "relation": "partner",
         "evidence": "no confidence key at all"},
    ]
    valid, n_bad = _validate_relationships(raw, universe, text)
    assert n_bad == 3
    assert [r["target_ticker"] for r in valid] == ["MSFT"]
    assert valid[0]["confidence"] == 0.9


def test_validate_relationships_still_filters_universe_and_type():
    universe = {"AAPL"}
    raw = [
        {"target_ticker": "ZZZZ", "relation": "supplier", "confidence": 0.9},
        {"target_ticker": "AAPL", "relation": "frenemy", "confidence": 0.9},
    ]
    valid, n_bad = _validate_relationships(raw, universe, "text")
    assert valid == []
    assert n_bad == 0  # out-of-universe/bad-type rows are not confidence drops


# --- evidence-substring check ---

def test_evidence_exact_substring_passes():
    text = "Apple supplies chips to Foxconn under a long-term agreement."
    assert _evidence_in_text("supplies chips to Foxconn", text)


def test_evidence_whitespace_mangled_passes_after_normalization():
    text = "Apple supplies\n   chips\t\tto\nFoxconn under a long-term agreement."
    assert _evidence_in_text("supplies chips to  Foxconn", text)
    # and the mirror case: mangled evidence against clean text
    assert _evidence_in_text("supplies\nchips   to Foxconn",
                             "Apple supplies chips to Foxconn today.")


def test_evidence_case_insensitive():
    assert _evidence_in_text("SUPPLIES CHIPS", "Apple supplies chips to Foxconn.")


def test_evidence_absent_fails():
    assert not _evidence_in_text("sells rockets to NASA",
                                 "Apple supplies chips to Foxconn.")


def test_evidence_empty_or_nonstring_fails():
    assert not _evidence_in_text("", "some text")
    assert not _evidence_in_text("   ", "some text")
    assert not _evidence_in_text(None, "some text")
    assert not _evidence_in_text(float("nan"), "some text")


def test_validate_relationships_sets_evidence_verified_flag():
    universe = {"MSFT", "NVDA"}
    text = "We compete   with\nMicrosoft in cloud services."
    raw = [
        {"target_ticker": "MSFT", "relation": "competitor", "confidence": 0.9,
         "evidence": "compete with Microsoft"},
        {"target_ticker": "NVDA", "relation": "supplier", "confidence": 0.8,
         "evidence": "NVIDIA supplies our GPUs"},  # not in text
    ]
    valid, _ = _validate_relationships(raw, universe, text)
    flags = {r["target_ticker"]: r["evidence_verified"] for r in valid}
    assert flags == {"MSFT": True, "NVDA": False}
    # failing rows are flagged, never dropped
    assert len(valid) == 2


# --- offline audit ---

@pytest.fixture
def fake_filings(tmp_path):
    """One fake 10-K on disk for ticker FAKE."""
    d = tmp_path / "FAKE"
    d.mkdir()
    filing = {
        "ticker": "FAKE",
        "company_name": "Fake Corp",
        "filing_date": "2024-03-15",
        "sections": {
            "Item 1 - Business": (
                "Fake Corp designs widgets. " * 10
                + "We purchase substantially all our gears from Acme Industrial. "
                + "Our principal competitor is Widgets Global Inc. " * 5
            ),
        },
    }
    (d / "10-K_2024-03-15_000000000001.json").write_text(json.dumps(filing))
    return tmp_path


def test_audit_evidence_report(fake_filings):
    df = pd.DataFrame({
        "source": ["FAKE", "FAKE", "FAKE", "MISS"],
        "target": ["ACME", "WGI", "XYZ", "ACME"],
        "relation": ["supplier", "competitor", "partner", "supplier"],
        "confidence": [0.9, 0.9, 0.9, 0.9],
        "evidence": [
            "purchase substantially  all our gears\nfrom Acme Industrial",  # mangled -> verifies
            "principal competitor is Widgets Global",                       # exact -> verifies
            "joint venture with XYZ Corp",                                  # absent -> fails
            "anything",                                                     # filing not on disk
        ],
        "filing_date": ["2024-03-15", "2024-03-15", "2024-03-15", "2024-03-15"],
    })  # no evidence_verified column: treated as un-verified, audit recomputes

    report = audit_evidence(df, filings_dir=fake_filings)
    assert report["n_filings_requested"] == 2
    assert report["n_filings_found"] == 1
    assert report["n_filings_missing"] == 1
    assert report["n_rows_checked"] == 4
    assert report["n_verified"] == 2
    assert report["hit_rate"] == pytest.approx(0.5)
    audited = report["rows"].set_index(["source", "target"])["evidence_verified_audit"]
    assert bool(audited[("FAKE", "ACME")]) is True   # whitespace-mangled, normalized
    assert bool(audited[("FAKE", "WGI")]) is True    # exact substring
    assert bool(audited[("FAKE", "XYZ")]) is False   # absent from filing
    assert bool(audited[("MISS", "ACME")]) is False  # filing not on disk


def test_audit_evidence_max_filings_limits_io(fake_filings):
    df = pd.DataFrame({
        "source": ["FAKE", "MISS"],
        "target": ["ACME", "ACME"],
        "relation": ["supplier", "supplier"],
        "confidence": [0.9, 0.9],
        "evidence": ["gears from Acme Industrial", "anything"],
        "filing_date": ["2024-03-15", "2024-03-15"],
    })
    report = audit_evidence(df, filings_dir=fake_filings, max_filings=1)
    assert report["n_filings_requested"] == 1
    assert report["n_rows_checked"] == 1
