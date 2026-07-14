"""Tests for C16 `insider_cluster_buy` (synthetic inputs only).

Pinned semantics under test (registration + protocol pin 2026-07-13, plus the
two ledgered build-time pins):
  - an event needs a SECOND DISTINCT qualifying owner: one owner never fires,
    the same owner twice never fires;
  - the 21-day window is between FILING dates, calendar, inclusive (a gap of
    exactly 21 days fires, 22 does not);
  - onset collapse: no re-fire until the trailing window drops below 2
    distinct owners;
  - entity exclusion per the implemented structural rule: tenpct-only owners
    (no officer/director flag) never qualify; amendments and sales never
    qualify;
  - credited-return window: exactly 63 trading days starting at
    pos(event) + credit_offset — offset 1 = the original implementation
    (credits the filing-day overnight close(d)->close(d+1)), offset 2 = the
    corrected headline timing per the ledgered entry-credit disambiguation
    (buy at close(d+1), first credit close(d+1)->close(d+2));
  - days with zero active names are 0.0 (cash) and stay in the series;
  - PIT: later filings change nothing earlier.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.signals.insider_cluster import (
    ORIGINAL_CREDIT_OFFSET,
    add_adv_diagnostic,
    build_events,
    build_series,
    compute,
    dropped_first_day_returns,
    qualifying_buys,
)


def _row(ticker="AAA", owner="O1", filing="2020-02-03", dollar=100_000.0,
         code="P", amend=False, officer=True, director=False, tenpct=False):
    return dict(ticker=ticker, owner_cik=owner,
                filing_date=pd.Timestamp(filing), dollar_value=dollar,
                trans_code=code, is_amendment=amend, is_officer=officer,
                is_director=director, is_tenpct=tenpct)


def _events(rows):
    return build_events(qualifying_buys(pd.DataFrame(rows)))


def _market(tickers=("AAA",), start="2020-01-01", days=220, rets=None,
            volume=1_000_000.0):
    """Business-day panel with exact constant daily returns per ticker."""
    cal = pd.bdate_range(start, periods=days)
    frames = []
    for i, tk in enumerate(tickers):
        r = 0.01 if rets is None else rets[i]
        close = 100.0 * np.cumprod(np.full(days, 1.0 + r))
        frames.append(pd.DataFrame(
            {"date": cal, "ticker": tk, "close": close, "volume": volume}))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Second-distinct-owner trigger
# --------------------------------------------------------------------------- #

def test_single_owner_never_fires():
    ev = _events([_row(owner="O1", filing="2020-02-03"),
                  _row(owner="O1", filing="2020-02-05"),
                  _row(owner="O1", filing="2020-02-10")])
    assert ev.empty


def test_second_distinct_owner_fires_at_second_filing_date():
    ev = _events([_row(owner="O1", filing="2020-02-03", dollar=100_000),
                  _row(owner="O2", filing="2020-02-13", dollar=200_000)])
    assert len(ev) == 1
    assert ev.loc[0, "event_date"] == pd.Timestamp("2020-02-13")
    assert ev.loc[0, "n_insiders"] == 2
    assert ev.loc[0, "sum_dollar"] == pytest.approx(300_000.0)


def test_same_day_distinct_owners_fire():
    ev = _events([_row(owner="O1", filing="2020-02-10"),
                  _row(owner="O2", filing="2020-02-10")])
    assert len(ev) == 1
    assert ev.loc[0, "event_date"] == pd.Timestamp("2020-02-10")


# --------------------------------------------------------------------------- #
# 21-calendar-day boundary
# --------------------------------------------------------------------------- #

def test_gap_of_exactly_21_days_fires():
    ev = _events([_row(owner="O1", filing="2020-02-03"),
                  _row(owner="O2", filing="2020-02-24")])
    assert len(ev) == 1
    assert ev.loc[0, "event_date"] == pd.Timestamp("2020-02-24")


def test_gap_of_22_days_does_not_fire():
    ev = _events([_row(owner="O1", filing="2020-02-03"),
                  _row(owner="O2", filing="2020-02-25")])
    assert ev.empty


# --------------------------------------------------------------------------- #
# Onset collapse
# --------------------------------------------------------------------------- #

def test_no_refire_while_window_still_clustered():
    ev = _events([_row(owner="O1", filing="2020-02-03"),
                  _row(owner="O2", filing="2020-02-10"),
                  _row(owner="O3", filing="2020-02-17")])
    assert len(ev) == 1
    assert ev.loc[0, "event_date"] == pd.Timestamp("2020-02-10")


def test_rearm_after_window_drops_below_two():
    ev = _events([_row(owner="O1", filing="2020-02-03", dollar=1.0),
                  _row(owner="O2", filing="2020-02-10", dollar=2.0),
                  # window empties (O1/O2 age out), O3 alone re-arms but
                  # cannot fire; O4 is the new second distinct owner
                  _row(owner="O3", filing="2020-03-10", dollar=4.0),
                  _row(owner="O4", filing="2020-03-16", dollar=8.0)])
    assert list(ev["event_date"]) == [pd.Timestamp("2020-02-10"),
                                      pd.Timestamp("2020-03-16")]
    assert ev.loc[1, "n_insiders"] == 2
    assert ev.loc[1, "sum_dollar"] == pytest.approx(12.0)


# --------------------------------------------------------------------------- #
# Entity / row exclusions (the implemented structural rule)
# --------------------------------------------------------------------------- #

def test_tenpct_only_owner_never_qualifies():
    ev = _events([_row(owner="O1", filing="2020-02-03"),
                  _row(owner="FUND", filing="2020-02-10",
                       officer=False, director=False, tenpct=True)])
    assert ev.empty


def test_tenpct_only_dollars_not_counted():
    ev = _events([_row(owner="O1", filing="2020-02-03", dollar=100.0),
                  _row(owner="FUND", filing="2020-02-10", dollar=1e9,
                       officer=False, director=False, tenpct=True),
                  _row(owner="O2", filing="2020-02-12", dollar=200.0,
                       officer=False, director=True)])
    assert len(ev) == 1
    assert ev.loc[0, "n_insiders"] == 2
    assert ev.loc[0, "sum_dollar"] == pytest.approx(300.0)


def test_officer_flagged_tenpct_still_qualifies():
    # the pinned structural rule only drops tenpct-ONLY owners
    ev = _events([_row(owner="O1", filing="2020-02-03"),
                  _row(owner="O2", filing="2020-02-10", tenpct=True)])
    assert len(ev) == 1


def test_amendments_and_sales_never_qualify():
    ev = _events([_row(owner="O1", filing="2020-02-03"),
                  _row(owner="O2", filing="2020-02-10", amend=True),
                  _row(owner="O3", filing="2020-02-11", code="S")])
    assert ev.empty


# --------------------------------------------------------------------------- #
# Portfolio: credited window (both timings), exit after 63 days, cash days
# --------------------------------------------------------------------------- #

def test_original_timing_credits_63_days_from_first_close():
    market = _market()
    cal = pd.DatetimeIndex(np.sort(market["date"].unique()))
    series, events = build_series(
        _events([_row(owner="O1", filing="2020-02-03"),
                 _row(owner="O2", filing="2020-02-10")]), market,
        credit_offset=ORIGINAL_CREDIT_OFFSET)
    assert bool(events.loc[0, "tradeable"])
    pos0 = cal.get_loc(pd.Timestamp("2020-02-10"))
    active = series["ret"].to_numpy()[pos0 + 1: pos0 + 64]
    assert len(active) == 63
    assert np.allclose(active, 0.01)
    assert series["ret"].iloc[pos0] == 0.0            # event day: not yet in
    assert series["ret"].iloc[pos0 + 64] == 0.0       # day 64: exited
    assert (series["n_active"].to_numpy()[pos0 + 1: pos0 + 64] == 1).all()


def test_corrected_timing_credits_63_days_from_second_close():
    # default = the corrected headline timing (credit_offset=2): the
    # filing-day overnight close(d)->close(d+1) is NOT credited
    market = _market()
    cal = pd.DatetimeIndex(np.sort(market["date"].unique()))
    series, _ = build_series(
        _events([_row(owner="O1", filing="2020-02-03"),
                 _row(owner="O2", filing="2020-02-10")]), market)
    pos0 = cal.get_loc(pd.Timestamp("2020-02-10"))
    active = series["ret"].to_numpy()[pos0 + 2: pos0 + 65]
    assert len(active) == 63
    assert np.allclose(active, 0.01)
    assert series["ret"].iloc[pos0 + 1] == 0.0        # dropped overnight day
    assert series["ret"].iloc[pos0 + 65] == 0.0       # day 65: exited
    assert (series["n_active"].to_numpy()[pos0 + 2: pos0 + 65] == 1).all()


def test_event_on_non_trading_day_enters_next_session():
    market = _market()
    ev = _events([_row(owner="O1", filing="2020-02-03"),
                  _row(owner="O2", filing="2020-02-08")])  # a Saturday
    # first trading day >= Sat 02-08 is Mon 02-10; original timing credits
    # from Tue 02-11, corrected timing from Wed 02-12
    orig, _ = build_series(ev, market, credit_offset=ORIGINAL_CREDIT_OFFSET)
    assert orig.loc[pd.Timestamp("2020-02-10"), "ret"] == 0.0
    assert orig.loc[pd.Timestamp("2020-02-11"), "ret"] == pytest.approx(0.01)
    corr, _ = build_series(ev, market)
    assert corr.loc[pd.Timestamp("2020-02-11"), "ret"] == 0.0
    assert corr.loc[pd.Timestamp("2020-02-12"), "ret"] == pytest.approx(0.01)


def test_days_without_active_names_are_cash_and_kept():
    market = _market(days=60)
    series, _ = build_series(_events([_row(owner="O1", filing="2020-02-03")]),
                             market)
    assert len(series) == 60                     # every calendar day present
    assert (series["ret"] == 0.0).all()
    assert (series["n_active"] == 0).all()


def test_equal_weight_across_active_names():
    market = _market(tickers=("AAA", "BBB"), rets=(0.01, 0.03))
    series, _ = build_series(
        _events([_row(ticker="AAA", owner="O1", filing="2020-02-03"),
                 _row(ticker="AAA", owner="O2", filing="2020-02-10"),
                 _row(ticker="BBB", owner="O3", filing="2020-02-03"),
                 _row(ticker="BBB", owner="O4", filing="2020-02-10")]),
        market)
    assert series.loc[pd.Timestamp("2020-02-12"), "ret"] == pytest.approx(0.02)
    assert series.loc[pd.Timestamp("2020-02-12"), "n_active"] == 2


def test_pre_calendar_event_is_not_tradeable():
    market = _market()
    series, events = build_series(
        _events([_row(owner="O1", filing="2019-01-07"),
                 _row(owner="O2", filing="2019-01-14")]), market)
    assert not bool(events.loc[0, "tradeable"])
    assert (series["ret"] == 0.0).all()


# --------------------------------------------------------------------------- #
# PIT perturbation
# --------------------------------------------------------------------------- #

def test_later_filings_change_nothing_earlier():
    market = _market(days=220)
    base = [_row(owner="O1", filing="2020-02-03"),
            _row(owner="O2", filing="2020-02-10")]
    later = [_row(owner="O3", filing="2020-06-01"),
             _row(owner="O4", filing="2020-06-08")]
    s1, e1 = compute(pd.DataFrame(base), market)
    s2, e2 = compute(pd.DataFrame(base + later), market)
    # the earlier event row is unchanged
    pd.testing.assert_frame_equal(e1.iloc[[0]], e2.iloc[[0]])
    # the series is unchanged through the later event's date (entry is t+1)
    cut = pd.Timestamp("2020-06-08")
    pd.testing.assert_frame_equal(s1.loc[:cut], s2.loc[:cut])
    assert len(e2) == 2


# --------------------------------------------------------------------------- #
# ADV diagnostic
# --------------------------------------------------------------------------- #

def test_adv_diagnostic_exact():
    market = _market(rets=(0.0,))                 # close 100, volume 1e6
    events = _events([_row(owner="O1", filing="2020-02-03", dollar=100_000),
                      _row(owner="O2", filing="2020-02-10", dollar=200_000)])
    _, events = build_series(events, market)
    out = add_adv_diagnostic(events, market)
    assert out.loc[0, "adv_20d"] == pytest.approx(1e8)
    assert out.loc[0, "dollar_over_adv"] == pytest.approx(3e-3)


# --------------------------------------------------------------------------- #
# Dropped-first-day decomposition
# --------------------------------------------------------------------------- #

def test_dropped_first_day_returns_exact():
    # constant +1%/day market: every event's dropped close(d)->close(d+1)
    # return is exactly 0.01
    market = _market()
    events = _events([_row(owner="O1", filing="2020-02-03"),
                      _row(owner="O2", filing="2020-02-10")])
    _, events = build_series(events, market)
    first = dropped_first_day_returns(events, market)
    assert len(first) == 1
    assert first.iloc[0] == pytest.approx(0.01)
