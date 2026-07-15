"""Monitoring front end: a read-only view of the research state.

Reads the repo's own artifacts — FACTORS.md, the pre-registration ledger, and
the parquet cache — and renders them as a self-contained HTML dashboard. It
computes no statistics and writes nothing except the dashboard file, so it can
never influence a result it displays.

    python -m alpha_graph.monitor           # build reports/dashboard.html + open
    python -m alpha_graph.monitor --serve   # localhost, rebuilt on each request
"""

from alpha_graph.monitor.dashboard import build, gather, render
from alpha_graph.monitor.health import scan_cache
from alpha_graph.monitor.ledger import load_ledger
from alpha_graph.monitor.registry import load_registry

__all__ = ["build", "gather", "render", "scan_cache", "load_ledger", "load_registry"]
