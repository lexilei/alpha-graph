"""Tests for the monitor front end (alpha_graph.monitor).

Fixtures are synthetic: the real FACTORS.md and ledger grow with every look,
so asserting against them would make these tests fail on unrelated research
commits. The one test that does read the live artifacts asserts only
structural invariants the project's own conventions guarantee.
"""

from datetime import date, datetime

import pandas as pd
import pytest

from alpha_graph.monitor.dashboard import (
    _bar_path,
    _fmt_t,
    _nice_ticks,
    gather,
    render,
)
from alpha_graph.monitor.health import inspect_parquet, scan_cache
from alpha_graph.monitor.ledger import load_ledger, parse_ledger
from alpha_graph.monitor.registry import load_registry, parse_factors_md

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

FACTORS_MD = """# Factor Registry

Some prose that must not be parsed as a row.

## Candidates (hypotheses under evaluation)

| ID | name | source | status | definition |
|----|------|--------|--------|------------|
| C1 | `cosine_similarity` | 10-K | rejected | TF-IDF cosine. Bar: sn incr \\|t\\| >= 1.5. |
| C11 | `evt8k_freq_z` | 8-K | candidate | Abnormal 8-K frequency. |
| C19 | `cust_mom_pc` | 10-K | registered | Parked. |

## Baseline (controls — not tested for promotion)

| ID | name | source | definition |
|----|------|--------|------------|
| B1 | `momentum_21d` | price | `close.pct_change(21)`. |

## Known issues (carry into any evaluation)

- **C1** — a bullet, not a row.
"""

LEDGER_MD = """# Factor Research Log

| When | What | Result |
|------|------|--------|
| 2026-07-13 | C11 evaluated vs BASELINE | sn incr t=−2.00 |
| 2026-07-14 | **Accounting refresh**: earlier row bringing the count to \
**N = 121, two-sided ceiling emax_null(242) = 2.83**. Standings: C16 +3.00 \
(candidate), C11 −2.00. 0 accepted. | recorded |
| 2026-07-15 | **Accounting refresh**: the latest row brings the count to \
**N = 149, two-sided ceiling emax_null(298) = 2.89** (5% family-wise threshold \
≈ 3.59). Standings unchanged in substance: C17 +2.01 (candidate, window-guard \
headline), C16 +3.33 (candidate, NOT treated as clearing — composition), \
C15 −2.34, C11 −2.02; C20, C21/C22 rejected. 0 accepted. | recorded |
"""


@pytest.fixture
def reg():
    return parse_factors_md(FACTORS_MD)


@pytest.fixture
def led():
    return parse_ledger(LEDGER_MD)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_registry_splits_both_tables(reg):
    assert [f.id for f in reg.candidates] == ["C1", "C11", "C19"]
    assert [f.id for f in reg.baseline] == ["B1"]


def test_registry_ignores_prose_and_bullets(reg):
    # "- **C1** — a bullet" sits under a third heading and must not become a row.
    assert len(reg.candidates) == 3


def test_registry_unescapes_pipes(reg):
    """The registry writes `\\|t\\|` constantly; splitting on a bare pipe would
    truncate the definition and shift every later cell."""
    c1 = reg.by_id("C1")
    assert "sn incr |t| >= 1.5" in c1.definition


def test_registry_strips_code_ticks(reg):
    assert reg.by_id("C11").name == "evt8k_freq_z"


def test_registry_baseline_has_no_status(reg):
    b1 = reg.by_id("B1")
    assert b1.status is None and b1.status_known and b1.group == "baseline"


def test_registry_status_counts(reg):
    assert reg.status_counts() == {"rejected": 1, "candidate": 1, "registered": 1}


def test_registry_flags_unknown_status():
    md = FACTORS_MD.replace("| 10-K | rejected |", "| 10-K | banana |")
    f = parse_factors_md(md).by_id("C1")
    assert f.status == "banana" and not f.status_known


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #

def test_ledger_latest_refresh_is_authoritative(led):
    """The ledger header pins the LAST accounting row as the truth; an earlier
    row stating N=121 must not win."""
    assert len(led.refreshes) == 2
    assert led.latest.n_looks == 149
    assert led.latest.ceiling == 2.89
    assert led.latest.familywise == 3.59
    assert led.latest.accepted == 0
    assert led.latest.day == date(2026, 7, 15)


def test_ledger_parses_unicode_minus_standings(led):
    s = {x.factor_id: x.t for x in led.latest.standings}
    assert s == {"C17": 2.01, "C16": 3.33, "C15": -2.34, "C11": -2.02}


def test_ledger_carries_the_not_clearing_note(led):
    """C16's +3.33 exceeds the 2.89 ceiling but is on record as NOT clearing.
    Losing this note would turn the dashboard into a false discovery claim."""
    c16 = next(s for s in led.latest.standings if s.factor_id == "C16")
    assert "NOT treated as clearing" in c16.note


def test_ledger_standings_ignore_the_narrative_preamble():
    """Only the segment after 'Standings:' is scanned — a factor id followed by
    a number in the preamble is not a standing."""
    md = LEDGER_MD.replace("the latest row brings", "C99 +9.99 rework brings")
    latest = parse_ledger(md).latest
    assert "C99" not in {s.factor_id for s in latest.standings}


def test_ledger_ceiling_cross_check(led):
    """The stated ceiling must reproduce from ic_tools.emax_null on the stated
    trial count — this is the check that catches a stale hand-typed number."""
    assert led.latest.ceiling_trials == 298
    assert led.latest.ceiling_drift() < 0.01


def test_ledger_ceiling_drift_is_detected():
    md = LEDGER_MD.replace("emax_null(298) = 2.89", "emax_null(298) = 1.99")
    assert parse_ledger(md).latest.ceiling_drift() > 0.5


def test_ledger_rows_and_recency(led):
    assert len(led.rows) == 3
    assert led.recent(2)[0].day == date(2026, 7, 15)  # newest first


def test_ledger_without_any_refresh():
    led = parse_ledger("| When | What | Result |\n| 2026-01-01 | a look | t=1 |\n")
    assert led.latest is None and len(led.rows) == 1


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

def test_inspect_parquet_reads_metadata(tmp_path):
    p = tmp_path / "x.parquet"
    pd.DataFrame({
        "date": pd.to_datetime(["2011-04-06", "2026-04-02", "2015-01-01"]),
        "ticker": ["AAPL", "MSFT", "AAPL"],
        "v": [1.0, 2.0, 3.0],
    }).to_parquet(p, index=False)

    e = inspect_parquet(p)
    assert e.rows == 3
    assert e.columns == ["date", "ticker", "v"]
    assert e.date_col == "date"
    assert e.date_range == "2011-04-06 → 2026-04-02"
    assert e.n_tickers is None          # shallow scan does not read columns
    assert inspect_parquet(p, deep=True).n_tickers == 2
    assert e.error is None


def test_inspect_parquet_prefers_availability_over_period(tmp_path):
    """A cache carrying both stamps spans the one the panel merges on."""
    p = tmp_path / "y.parquet"
    pd.DataFrame({"end": pd.to_datetime(["2020-01-01"]),
                  "filed": pd.to_datetime(["2020-03-01"])}).to_parquet(p, index=False)
    assert inspect_parquet(p).date_col == "filed"


def test_inspect_parquet_dateless_is_not_an_error(tmp_path):
    p = tmp_path / "z.parquet"
    pd.DataFrame({"sharpe": [0.8]}).to_parquet(p, index=False)
    e = inspect_parquet(p)
    assert e.date_col is None and e.date_range is None and e.error is None


def test_inspect_parquet_reports_corruption(tmp_path):
    p = tmp_path / "bad.parquet"
    p.write_bytes(b"not a parquet file")
    e = inspect_parquet(p)
    assert e.error is not None and e.rows == 0


def test_scan_cache_flags_pre_reference_writes(tmp_path):
    old, ref = tmp_path / "old.parquet", tmp_path / "market_data.parquet"
    for f in (old, ref):
        pd.DataFrame({"a": [1]}).to_parquet(f, index=False)
    import os
    os.utime(old, (1_000_000, 1_000_000))       # long before the reference

    rep = scan_cache(tmp_path)
    assert rep.reference is not None
    assert [e.name for e in rep.older_than_reference()] == ["old.parquet"]
    assert rep.failed() == []


def test_scan_cache_missing_dir(tmp_path):
    rep = scan_cache(tmp_path / "nope")
    assert rep.entries == [] and rep.reference is None


# --------------------------------------------------------------------------- #
# Dashboard primitives
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lo,hi,count,want", [
    (0, 4.09, 4, [0.0, 1.0, 2.0, 3.0, 4.0]),       # the standings axis
    (2.815, 2.905, 3, [2.825, 2.85, 2.875, 2.9]),  # sub-1 magnitude: the bug
    (5, 5, 3, [5]),                                # degenerate span
])
def test_nice_ticks(lo, hi, count, want):
    assert _nice_ticks(lo, hi, count) == want


def test_nice_ticks_never_returns_a_single_tick_on_a_real_span():
    """A one-tick axis is unreadable; it was the ceiling chart's first render."""
    for hi in (2.905, 3.5, 120.0, 0.004):
        assert len(_nice_ticks(hi * 0.97, hi, 3)) >= 2


def test_bar_path_is_square_at_the_baseline():
    d = _bar_path(10, 0, 100, 20, r=4)
    assert d.startswith("M10,0")      # grows from the baseline
    assert d.endswith("H10 Z")
    assert "A4,4" in d                # rounded data-end


def test_bar_path_degenerates_below_the_radius():
    assert "A" not in _bar_path(0, 0, 1, 20, r=4)


def test_fmt_t_uses_the_ledgers_minus_sign():
    assert _fmt_t(2.01) == "+2.01"
    assert _fmt_t(-2.34) == "−2.34"   # U+2212, not a hyphen


# --------------------------------------------------------------------------- #
# End-to-end against the live repo artifacts (structural invariants only)
# --------------------------------------------------------------------------- #

def test_live_registry_parses():
    reg = load_registry()
    assert len(reg.candidates) >= 24
    assert all(f.status_known for f in reg.candidates), \
        [(f.id, f.status) for f in reg.candidates if not f.status_known]
    assert all(f.name and f.id.startswith("C") for f in reg.candidates)
    assert all(f.id.startswith("B") for f in reg.baseline)


def test_live_ledger_ceiling_matches_ic_tools():
    """Guards the ledger itself: every accounting row's stated ceiling must
    reproduce from emax_null on its own stated trial count."""
    for r in load_ledger().refreshes:
        assert r.ceiling_drift() < 0.01, f"{r.day}: stated {r.ceiling}"


def test_live_ledger_standings_have_registry_rows():
    reg, led = load_registry(), load_ledger()
    for s in led.latest.standings:
        assert reg.by_id(s.factor_id) is not None, s.factor_id


def test_render_produces_a_self_contained_page():
    html = render(gather())
    assert html.startswith("<!doctype html>")
    assert "<script src=" not in html and "http://" not in html.replace(
        "http://127.0.0.1", "")          # no external fetches
    assert "alpha-graph monitor" in html


def test_render_never_drops_a_standing_note():
    """Whatever the ledger adjudicates travels into the page."""
    ctx = gather()
    html = render(ctx)
    for s in ctx.ledger.latest.standings:
        assert s.factor_id in html
        if s.note:
            assert s.note.split(",")[0] in html


def test_gather_is_read_only(tmp_path):
    """The monitor must not be able to influence a result it displays."""
    before = {p: p.stat().st_mtime for p in
              (load_registry().source_path, load_ledger().source_path)}
    gather()
    assert {p: p.stat().st_mtime for p in before} == before


def test_context_datetime_is_stamped():
    assert isinstance(gather().built_at, datetime)
