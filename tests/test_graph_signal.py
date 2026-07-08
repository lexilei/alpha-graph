"""Unit tests for the knowledge graph spillover signal."""

import math

import networkx as nx
import pandas as pd
import pytest

from alpha_graph.signals.graph_signal import (
    EDGE_WEIGHTS,
    compute_neighbor_signal,
)


def _make_graph() -> nx.DiGraph:
    """Build a simple test graph.

    AAPL --supplier--> MSFT (confidence 0.9)
    AAPL --customer--> NVDA (confidence 0.8)
    GOOG --competitor-> MSFT (confidence 0.7)
    AMZN --partner---> AAPL (confidence 0.6)
    """
    G = nx.DiGraph()
    G.add_nodes_from(["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "ISOLATED"])
    G.add_edge("AAPL", "MSFT", relation="supplier", confidence=0.9)
    G.add_edge("AAPL", "NVDA", relation="customer", confidence=0.8)
    G.add_edge("GOOG", "MSFT", relation="competitor", confidence=0.7)
    G.add_edge("AMZN", "AAPL", relation="partner", confidence=0.6)
    return G


def test_edge_weights_cover_all_types():
    expected = {"supplier", "customer", "competitor", "partner"}
    assert set(EDGE_WEIGHTS.keys()) == expected


def test_competitor_weight_is_negative():
    assert EDGE_WEIGHTS["competitor"] < 0


def test_neighbor_signal_basic():
    G = _make_graph()
    signals = {"MSFT": 0.5, "NVDA": -0.3, "AMZN": 0.1}

    result = compute_neighbor_signal(G, "AAPL", signals)

    # AAPL has out-edges to MSFT (supplier) and NVDA (customer)
    # and in-edge from AMZN (partner)
    # All should contribute
    assert result != 0.0


def test_neighbor_signal_no_neighbors():
    """Unconnected node -> NaN (not 0: zero is a real signal value)."""
    G = _make_graph()
    signals = {"AAPL": 0.5, "MSFT": -0.2}

    result = compute_neighbor_signal(G, "ISOLATED", signals)
    assert math.isnan(result)


def test_neighbor_signal_no_signal_values():
    G = _make_graph()
    # No signal values for any of AAPL's neighbors
    signals = {"ISOLATED": 0.5}

    result = compute_neighbor_signal(G, "AAPL", signals)
    assert math.isnan(result)


def test_neighbor_signal_competitor_inverts():
    """Competitor's bad news should produce positive spillover."""
    G = nx.DiGraph()
    G.add_nodes_from(["A", "B"])
    G.add_edge("A", "B", relation="competitor", confidence=1.0)

    # B has negative signal (bad news)
    signals = {"B": -0.5}
    result = compute_neighbor_signal(G, "A", signals)

    # Competitor weight is negative, so negative * negative = positive
    assert result > 0


def test_neighbor_signal_supplier_propagates():
    """Supplier's bad news should propagate negatively."""
    G = nx.DiGraph()
    G.add_nodes_from(["A", "B"])
    G.add_edge("A", "B", relation="supplier", confidence=1.0)

    signals = {"B": -0.5}
    result = compute_neighbor_signal(G, "A", signals)

    # Supplier weight is positive, negative signal stays negative
    assert result < 0


def test_neighbor_signal_missing_ticker():
    """Ticker not in graph should return NaN."""
    G = _make_graph()
    signals = {"AAPL": 0.5}

    result = compute_neighbor_signal(G, "NOT_IN_GRAPH", signals)
    assert math.isnan(result)


def test_neighbor_signal_in_edge_relation_reversed():
    """In-edge relations are stated from the SOURCE's perspective and must flip.

    S --supplier--> X means "X is S's supplier", so from X's side S is a
    customer (weight 0.8), not a supplier (1.0). A second in-edge of a
    self-symmetric type (partner) makes the normalized average detect the flip.
    """
    G = nx.DiGraph()
    G.add_nodes_from(["X", "S", "P"])
    G.add_edge("S", "X", relation="supplier", confidence=1.0)
    G.add_edge("P", "X", relation="partner", confidence=1.0)

    signals = {"S": 1.0, "P": -1.0}
    result = compute_neighbor_signal(G, "X", signals)

    # reversed: customer(0.8)*1 + partner(0.5)*(-1) = 0.3, weight total 1.3
    assert result == pytest.approx(0.3 / 1.3)
    # the unreversed (buggy) value would be 0.5 / 1.5
    assert result != pytest.approx(0.5 / 1.5)


def test_load_event_scores_pit_never_uses_static(tmp_path, monkeypatch):
    """With as_of_date set and no timeseries cache, the dateless static
    snapshot must NOT be used — it would leak today's scores into the past."""
    import alpha_graph.signals.graph_signal as gs

    static = pd.DataFrame({"ticker": ["AAPL"], "event_score": [0.7]})
    static.to_parquet(tmp_path / "event_signals.parquet", index=False)
    monkeypatch.setattr(gs, "CACHE_DIR", tmp_path)

    assert gs._load_event_scores(pd.Timestamp("2020-01-31")) == {}
    # without a date the static snapshot is still the intended source
    assert gs._load_event_scores(None) == {"AAPL": 0.7}


def test_build_graph_respects_as_of_date():
    from alpha_graph.data.relationships import build_graph

    # Create test relationships with different dates
    rels = pd.DataFrame([
        {"source": "AAPL", "target": "MSFT", "relation": "supplier",
         "confidence": 0.9, "evidence": "test", "filing_date": "2024-01-01"},
        {"source": "GOOG", "target": "NVDA", "relation": "customer",
         "confidence": 0.8, "evidence": "test", "filing_date": "2025-06-01"},
    ])
    rels["filing_date"] = pd.to_datetime(rels["filing_date"])

    # As of 2024-06-01: only the first edge should exist
    G = build_graph(rels, as_of_date="2024-06-01")
    assert G.has_edge("AAPL", "MSFT")
    assert not G.has_edge("GOOG", "NVDA")

    # As of 2026-01-01: both edges should exist
    G2 = build_graph(rels, as_of_date="2026-01-01")
    assert G2.has_edge("AAPL", "MSFT")
    assert G2.has_edge("GOOG", "NVDA")


def test_build_graph_deduplicates():
    from alpha_graph.data.relationships import build_graph

    # Same edge from two filings — should keep the newer one
    rels = pd.DataFrame([
        {"source": "AAPL", "target": "MSFT", "relation": "supplier",
         "confidence": 0.5, "evidence": "old", "filing_date": "2023-01-01"},
        {"source": "AAPL", "target": "MSFT", "relation": "partner",
         "confidence": 0.9, "evidence": "new", "filing_date": "2024-01-01"},
    ])
    rels["filing_date"] = pd.to_datetime(rels["filing_date"])

    G = build_graph(rels)
    # Should have only 1 edge (deduplicated), with the newer relation
    assert G.number_of_edges() == 1
    edge_data = G.edges["AAPL", "MSFT"]
    assert edge_data["relation"] == "partner"
    assert edge_data["confidence"] == 0.9
