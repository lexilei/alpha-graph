"""Tests for the binary-book risk metrics.

Where possible these drive the *recorded production session*
(2026-07-25 Kalshi FV-maker) rather than invented fixtures, because the
defect being fixed was invisible to 25 rounds of review that all reasoned
inside the wrong model.  The decisive test is the regression at the bottom:
the metric the maker actually used passes at a moment when the account was
one $50 BTC move from zero.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alpha_graph.trading.binary_risk import (
    HighWaterBudget,
    Region,
    _payout,
    gross_contracts,
    parse_exchange_ts,
    region_from_strikes,
    worst_case_payoff,
)

FIXTURES = Path(__file__).parent / "fixtures"
STRIKES = FIXTURES / "kalshi_btc_strikes_2026-07-25.json"

# The book held at 2026-07-25 20:02:47Z, straight out of the telemetry.
# Marked equity read $57.64; cash was $0.0599.
REAL_BOOK = {
    "KXBTC-26JUL2517-B64125": -30.0,
    "KXBTC-26JUL2517-B64375": +5.0,
    "KXBTCD-26JUL2517-T63999.99": -14.0,
    "KXBTCD-26JUL2517-T64249.99": +35.0,
}
REAL_REGIONS = {
    "KXBTC-26JUL2517-B64125": Region(64000.0, 64249.99),
    "KXBTC-26JUL2517-B64375": Region(64250.0, 64499.99),
    "KXBTCD-26JUL2517-T63999.99": Region(63999.99),
    "KXBTCD-26JUL2517-T64249.99": Region(64249.99),
}


# --------------------------------------------------------------------- Region

def test_region_is_half_open_matching_the_fv_formula():
    r = Region(64000.0, 64249.99)
    assert not r.pays(64000.0)      # (lo, hi] excludes the floor
    assert r.pays(64000.01)
    assert r.pays(64249.99)         # ...and includes the cap
    assert not r.pays(64250.0)


def test_region_rejects_empty_intervals():
    with pytest.raises(ValueError):
        Region(64250.0, 64000.0)
    with pytest.raises(ValueError):
        Region(64000.0, 64000.0)


def test_region_from_strikes_covers_all_three_kalshi_shapes():
    # greater / between / less, the strike_type values the exchange ships
    assert region_from_strikes(64249.99) == Region(64249.99, math.inf)
    assert region_from_strikes(64000, 64249.99) == Region(64000.0, 64249.99)
    assert region_from_strikes(64000, "") == Region(64000.0, math.inf)
    assert region_from_strikes(None, 54700) == Region(-math.inf, 54700.0)


def test_less_market_pays_at_or_below_its_cap():
    r = region_from_strikes(None, 54700)
    assert r.pays(54700.0) and r.pays(1.0)
    assert not r.pays(54700.01)
    # and it carries real risk rather than being written off as unknown
    wc = worst_case_payoff({"a": -10.0}, {"a": r})
    assert wc.value == 0.0 and wc.at_price <= 54700.0


@pytest.mark.parametrize("floor,cap", [(None, None), ("", ""), ("abc", None),
                                       (64000, "abc"), (None, "abc"),
                                       (float("nan"), None), (64000, 63000)])
def test_region_from_strikes_refuses_to_invent_geometry(floor, cap):
    """Absent is a shape; unparseable is a fault.

    A bracket whose cap fails to parse must NOT degrade to an unbounded
    threshold -- that widens the payout region and makes the risk gate
    optimistic, the one direction it may never fail in.
    """
    assert region_from_strikes(floor, cap) is None


# ------------------------------------------------------------- worst case

def test_long_yes_can_pay_nothing():
    wc = worst_case_payoff({"a": 10.0}, {"a": Region(100.0, 200.0)})
    assert wc.value == 0.0
    assert not Region(100.0, 200.0).pays(wc.at_price)


def test_short_yes_is_long_no_and_pays_when_the_region_misses():
    # a lone short can only fail to pay inside its own region
    wc = worst_case_payoff({"a": -10.0}, {"a": Region(100.0, 200.0)})
    assert wc.value == 0.0
    assert Region(100.0, 200.0).pays(wc.at_price)


def test_disjoint_brackets_net_to_zero_but_can_both_lose():
    """The exact defect: net = 0 reads flat, the book can still pay nothing."""
    pos = {"lo": +10.0, "hi": -10.0}
    regions = {"lo": Region(100.0, 200.0), "hi": Region(200.0, 300.0)}
    assert sum(pos.values()) == 0.0            # the metric the maker used
    wc = worst_case_payoff(pos, regions)
    assert wc.value == 0.0                     # the metric it needed
    assert 200.0 < wc.at_price <= 300.0        # both legs dead here


def test_offsetting_positions_in_the_same_region_are_genuinely_flat():
    pos = {"a": +10.0, "b": -10.0}
    regions = {"a": Region(100.0, 200.0), "b": Region(100.0, 200.0)}
    assert worst_case_payoff(pos, regions).value == 10.0


def test_unknown_region_is_assumed_to_pay_nothing():
    known = worst_case_payoff({"a": -10.0}, {"a": Region(100.0, 200.0)})
    plus_unknown = worst_case_payoff({"a": -10.0, "b": 7.0}, {"a": Region(100.0, 200.0)})
    assert plus_unknown.n_unknown == 1
    assert plus_unknown.value <= known.value   # never optimistic


def test_zero_positions_are_ignored():
    wc = worst_case_payoff({"a": 0.0, "b": 0.0}, {})
    assert wc.value == 0.0 and wc.n_unknown == 0


def test_empty_book_is_riskless():
    assert worst_case_payoff({}, {}).value == 0.0


def test_gross_contracts():
    assert gross_contracts({"a": -30.0, "b": 5.0}) == 35.0


# ------------------------------------------- the recorded production book

def test_recorded_book_settles_to_zero_somewhere():
    """2026-07-25 20:02:47Z. Marked equity $57.64, cash $0.0599.

    There is a settlement price at which all four legs pay nothing, so the
    true floor under the account was $0.06.
    """
    wc = worst_case_payoff(REAL_BOOK, REAL_REGIONS)
    assert wc.value == 0.0
    assert 64100.0 < wc.at_price <= 64249.99
    for tick, q in REAL_BOOK.items():
        assert _payout(q, REAL_REGIONS[tick], wc.at_price) == 0.0


def test_recorded_book_defeats_the_net_position_gate():
    """The regression that cost $43.72: the old gate sees a tame book."""
    net = sum(REAL_BOOK.values())
    assert abs(net) <= 20                       # NET_CAP = 20 -> passes
    cash = 0.0599
    risk_equity = cash + worst_case_payoff(REAL_BOOK, REAL_REGIONS).value
    assert risk_equity < 1.0                    # ...on a $49.53 account


def test_probe_set_is_exhaustive_against_brute_force():
    """The analytic probe set must equal a fine grid search, everywhere."""
    regions = dict(REAL_REGIONS)
    lo = min(r.lo for r in regions.values()) - 500
    hi = max(r.hi for r in regions.values() if math.isfinite(r.hi)) + 500
    n = 20000
    grid = [lo + i * (hi - lo) / n for i in range(n + 1)]
    for book in (REAL_BOOK,
                 {k: -v for k, v in REAL_BOOK.items()},
                 {"KXBTC-26JUL2517-B64125": 4.0}):
        analytic = worst_case_payoff(book, regions).value
        brute = min(sum(_payout(q, regions.get(t), s) for t, q in book.items())
                    for s in grid)
        assert analytic == pytest.approx(brute, abs=1e-9)


@pytest.mark.skipif(not STRIKES.exists(), reason="strike fixture absent")
def test_every_session_strike_parses():
    blob = json.loads(STRIKES.read_text())
    assert len(blob) >= 28
    for tick, v in blob.items():
        reg = region_from_strikes(v["floor"], v["cap"])
        assert reg is not None, tick
        # the recorded outcome must be consistent with the region being a
        # real interval on price, not a mislabelled field
        assert reg.lo < reg.hi


# ------------------------------------------------------------ loss budget

def test_high_water_ratchets_and_never_falls():
    b = HighWaterBudget(budget=15.0)
    assert b.floor() is None and not b.breached(0.0)   # unarmed: cannot fire
    b.observe(49.53)
    assert b.floor() == pytest.approx(34.53)
    b.observe(70.0)
    b.observe(20.0)                                    # a drawdown
    assert b.high_water == 70.0                        # does not un-ratchet
    assert b.floor() == pytest.approx(55.0)


def test_budget_does_not_reset_on_a_new_day():
    """The 00:05Z re-arm: a fresh $15 on an account already down 49%.

    There is no date in this object, so there is nothing for a calendar roll
    to reset.  Reloading yesterday's blob reproduces yesterday's floor.
    """
    b = HighWaterBudget(budget=15.0)
    b.observe(49.53)
    b.observe(34.27)                       # where the real stop fired
    reloaded = HighWaterBudget.load(15.0, b.serialise())
    assert reloaded.floor() == pytest.approx(34.53)
    assert reloaded.breached(25.43)        # the equity it re-armed on


def test_budget_ignores_junk_and_nonfinite():
    assert HighWaterBudget.load(15.0, None).high_water is None
    assert HighWaterBudget.load(15.0, "").high_water is None
    assert HighWaterBudget.load(15.0, "garbage").high_water is None
    assert HighWaterBudget.load(15.0, "inf").high_water is None
    b = HighWaterBudget(budget=15.0)
    b.observe(float("nan"))
    b.observe(float("inf"))
    assert b.high_water is None


def test_budget_headroom():
    b = HighWaterBudget(budget=15.0)
    assert b.headroom(10.0) == math.inf
    b.observe(50.0)
    assert b.headroom(40.0) == pytest.approx(5.0)


# ------------------------------------------------------------- timestamps

@pytest.mark.parametrize("raw,micro", [
    ("2026-07-26T00:21:59.143172Z", 143172),
    ("2026-07-26T00:09:00.04888Z", 48880),
    ("2026-07-25T22:37:18.5932Z", 593200),
    ("2026-07-25T22:37:18.148Z", 148000),
    ("2026-07-25T22:37:18Z", 0),
    ("2026-07-25T22:37:18.1431728Z", 143172),   # 7 digits: truncate, don't crash
])
def test_parse_exchange_ts_handles_every_observed_width(raw, micro):
    dt = parse_exchange_ts(raw)
    assert dt is not None and dt.microsecond == micro
    assert dt.tzinfo is timezone.utc


@pytest.mark.parametrize("raw", [None, "", "not a time", "2026-13-45T99:99:99Z"])
def test_parse_exchange_ts_rejects_junk(raw):
    assert parse_exchange_ts(raw) is None


def test_string_compare_hazard_is_real_and_parsing_fixes_it():
    """A fill with no fractional part sorts *after* one with, as a string.

    'Z' (0x5A) > '.' (0x2E), so the maker's raw-string watermark would jump
    past the fractional fills of that second and drop them.
    """
    earlier = "2026-07-26T00:21:59.148000Z"
    later = "2026-07-26T00:21:59Z"
    assert later > earlier                                   # the hazard
    assert parse_exchange_ts(later) < parse_exchange_ts(earlier)   # the truth


def test_parse_exchange_ts_offset_form():
    assert parse_exchange_ts("2026-07-26T00:00:00+00:00") == datetime(
        2026, 7, 26, tzinfo=timezone.utc)
    assert parse_exchange_ts("2026-07-25T20:00:00-04:00") == datetime(
        2026, 7, 26, tzinfo=timezone.utc)
