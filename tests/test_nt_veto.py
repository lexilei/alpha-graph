"""Tests for the C18 nt_filing_veto construction (synthetic inputs only).

Pinned rules under test (FACTORS.md C18; protocol row 2026-07-13):
  - first-or-rare: a 2nd NT within 3 years does NOT trigger (boundary: an NT
    exactly 3 years earlier still blocks); 3 years + 1 day later it does;
    blocked filings still count as history for later ones; same-day rows
    collapse to one trigger;
  - the veto window is exactly 126 trading days, clipped at the calendar end;
  - availability is filing_date + 1 CALENDAR day (a Friday filing becomes
    effective Monday; a mid-week filing the next trading day, never same-day);
  - the exclusion removes a name at the NEXT rebalance only — a window opening
    mid-month never touches the already-held book;
  - no NT filings => WITH and WITHOUT books are identical;
  - the EW book is buy-and-hold within the month (drifting weights), halted
    days forward-fill.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.signals.nt_veto import (
    LOOKBACK_YEARS,
    WINDOW_TDAYS,
    book_holdings,
    build_veto,
    ew_book_returns,
    excluded_at,
    first_or_rare,
    veto_windows,
)


def _nt(ticker, date, accession, acceptance=None):
    d = pd.Timestamp(date)
    return dict(ticker=ticker, filing_date=d, accession=accession,
                acceptance_ts=pd.Timestamp(acceptance) if acceptance
                else d + pd.Timedelta(hours=17))


def _frame(rows):
    cols = ["ticker", "filing_date", "accession", "acceptance_ts"]
    return pd.DataFrame(rows, columns=cols)


CAL = pd.bdate_range("2020-01-01", "2021-12-31")


# --------------------------------------------------------------------------- #
# first-or-rare
# --------------------------------------------------------------------------- #

def test_pinned_constants():
    assert LOOKBACK_YEARS == 3
    assert WINDOW_TDAYS == 126


def test_first_nt_triggers_second_within_3y_does_not():
    nt = _frame([_nt("A", "2015-01-10", "a1"),
                 _nt("A", "2018-01-09", "a2")])   # 3y - 1d after the first
    trig = first_or_rare(nt)
    assert list(trig["accession"]) == ["a1"]


def test_second_nt_exactly_3y_later_still_blocked():
    nt = _frame([_nt("C", "2015-01-10", "c1"),
                 _nt("C", "2018-01-10", "c2")])   # exactly 3y: still in [t-3y, t)
    assert list(first_or_rare(nt)["accession"]) == ["c1"]


def test_second_nt_after_3y_plus_1d_triggers():
    nt = _frame([_nt("B", "2015-01-10", "b1"),
                 _nt("B", "2018-01-11", "b2")])   # 3y + 1d after the first
    assert list(first_or_rare(nt)["accession"]) == ["b1", "b2"]


def test_blocked_filing_still_counts_as_history():
    nt = _frame([_nt("A", "2015-01-10", "a1"),
                 _nt("A", "2016-01-10", "a2"),    # blocked by a1
                 _nt("A", "2018-06-01", "a3")])   # a1 out of window, a2 blocks
    assert list(first_or_rare(nt)["accession"]) == ["a1"]


def test_same_day_rows_collapse_to_one_trigger():
    nt = _frame([_nt("D", "2020-02-03", "d2", acceptance="2020-02-03 17:00"),
                 _nt("D", "2020-02-03", "d1", acceptance="2020-02-03 06:00")])
    trig = first_or_rare(nt)
    assert len(trig) == 1
    assert trig["accession"].iloc[0] == "d1"     # earliest acceptance wins


def test_rule_is_per_ticker():
    nt = _frame([_nt("A", "2020-02-03", "a1"),
                 _nt("B", "2020-06-01", "b1")])   # different firm: no blocking
    assert len(first_or_rare(nt)) == 2


# --------------------------------------------------------------------------- #
# window construction
# --------------------------------------------------------------------------- #

def test_window_exactly_126_trading_days():
    nt = _frame([_nt("A", "2020-02-03", "a1")])   # Monday
    w = veto_windows(first_or_rare(nt), CAL)
    assert len(w) == 1
    r = w.iloc[0]
    assert r["veto_from"] == pd.Timestamp("2020-02-04")   # +1cd, a trading day
    i0 = CAL.get_loc(r["veto_from"])
    assert r["veto_to"] == CAL[i0 + WINDOW_TDAYS - 1]
    assert len(CAL[(CAL >= r["veto_from"]) & (CAL <= r["veto_to"])]) == 126
    assert r["trigger_accession"] == "a1"


def test_window_clipped_at_calendar_end():
    nt = _frame([_nt("A", "2021-11-30", "a1")])
    w = veto_windows(first_or_rare(nt), CAL)
    assert w.iloc[0]["veto_to"] == CAL[-1]
    assert len(CAL[(CAL >= w.iloc[0]["veto_from"]) & (CAL <= w.iloc[0]["veto_to"])]) < 126


def test_pre_calendar_trigger_dropped_and_post_calendar_ignored():
    nt = _frame([_nt("A", "2019-12-15", "a1"),    # avail before CAL[0]
                 _nt("B", "2021-12-31", "b1")])   # avail after CAL[-1]
    w = veto_windows(first_or_rare(nt), CAL)
    assert len(w) == 0


def test_availability_is_next_calendar_day():
    # Friday filing -> available Saturday -> first trading day Monday.
    nt = _frame([_nt("A", "2020-03-06", "a1")])
    w = veto_windows(first_or_rare(nt), CAL)
    assert w.iloc[0]["veto_from"] == pd.Timestamp("2020-03-09")
    # Wednesday filing -> Thursday, never the same close.
    nt = _frame([_nt("B", "2020-03-04", "b1")])
    w = veto_windows(first_or_rare(nt), CAL)
    assert w.iloc[0]["veto_from"] == pd.Timestamp("2020-03-05")


def test_excluded_at_bounds_inclusive():
    w = pd.DataFrame([{"ticker": "A", "veto_from": pd.Timestamp("2020-02-04"),
                       "veto_to": pd.Timestamp("2020-08-04"), "trigger_accession": "a1"}])
    assert excluded_at(w, pd.Timestamp("2020-02-03")) == frozenset()
    assert excluded_at(w, pd.Timestamp("2020-02-04")) == frozenset({"A"})
    assert excluded_at(w, pd.Timestamp("2020-08-04")) == frozenset({"A"})
    assert excluded_at(w, pd.Timestamp("2020-08-05")) == frozenset()


# --------------------------------------------------------------------------- #
# EW book and rebalance-only exclusion
# --------------------------------------------------------------------------- #

def _two_name_market():
    cal = pd.bdate_range("2020-01-01", "2020-06-30")
    n = np.arange(len(cal))
    close = pd.DataFrame({"A": 100 * 1.001 ** n, "B": 100 * 0.999 ** n}, index=cal)
    month_ends = [g.max() for _, g in cal.to_series().groupby([cal.year, cal.month])]
    rebalances = [cal[0]] + month_ends[:-2]      # forming dates; end = next month-end
    end = month_ends[-2]
    return cal, close, rebalances, end


def test_exclusion_at_next_rebalance_only_not_mid_month():
    cal, close, rebalances, end = _two_name_market()
    members_at = lambda d: frozenset({"A", "B"})  # noqa: E731
    nt = _frame([_nt("B", "2020-01-15", "b1")])   # mid-month NT
    windows = build_veto(nt, cal)

    h_with = book_holdings(close, rebalances, members_at)
    h_without = book_holdings(close, rebalances, members_at, windows)
    jan_end = pd.Timestamp("2020-01-31")
    assert h_without[rebalances[0]] == ["A", "B"]     # window not yet open on 01-01
    assert h_without[jan_end] == ["A"]                # excluded at the NEXT rebalance
    assert all(h_with[d] == ["A", "B"] for d in rebalances)

    r_with = ew_book_returns(close, rebalances, h_with, end)
    r_without = ew_book_returns(close, rebalances, h_without, end)
    pre = r_with.index <= jan_end                     # held book untouched mid-month
    pd.testing.assert_series_equal(r_with[pre], r_without[pre])
    post = r_without.index > jan_end                  # from Feb the book is A alone
    assert np.allclose(r_without[post], 0.001)
    assert not np.allclose(r_with[post], 0.001)


def test_expired_window_readmits_at_rebalance():
    cal, close, rebalances, end = _two_name_market()
    members_at = lambda d: frozenset({"A", "B"})  # noqa: E731
    nt = _frame([_nt("B", "2020-01-15", "b1")])
    windows = veto_windows(first_or_rare(nt), cal, window_tdays=10)  # ends 01-29
    h = book_holdings(close, rebalances, members_at, windows)
    assert h[pd.Timestamp("2020-01-31")] == ["A", "B"]


def test_no_nt_books_identical():
    cal, close, rebalances, end = _two_name_market()
    members_at = lambda d: frozenset({"A", "B"})  # noqa: E731
    windows = build_veto(_frame([]), cal)
    assert len(windows) == 0
    h_with = book_holdings(close, rebalances, members_at)
    h_without = book_holdings(close, rebalances, members_at, windows)
    assert h_with == h_without
    pd.testing.assert_series_equal(
        ew_book_returns(close, rebalances, h_with, end),
        ew_book_returns(close, rebalances, h_without, end))


def test_buy_and_hold_weights_drift_and_halt_ffills():
    cal, close, rebalances, end = _two_name_market()
    close = close.copy()
    halt = cal[(cal >= "2020-02-10") & (cal <= "2020-02-11")]
    close.loc[halt, "B"] = np.nan                 # 2-day halt mid-February
    h = {d: ["A", "B"] for d in rebalances}
    r = ew_book_returns(close, rebalances, h, end)
    assert not r.isna().any()
    # Buy-and-hold second day of a segment: W2 = (1.001^2 + 0.999^2)/2,
    # W1 = (1.001 + 0.999)/2 = 1 -> r2 = W2 - 1 (not the EW-daily mean 0).
    d2 = r.index[1]
    expected = (1.001 ** 2 + 0.999 ** 2) / 2 - 1
    assert r.loc[d2] == pytest.approx(expected, rel=1e-12)


def test_ew_book_fails_closed():
    cal, close, rebalances, end = _two_name_market()
    with pytest.raises(ValueError, match="empty book"):
        ew_book_returns(close, rebalances, {d: [] for d in rebalances}, end)
    close.loc[rebalances[1], "B"] = np.nan
    with pytest.raises(ValueError, match="no close at forming date"):
        ew_book_returns(close, rebalances, {d: ["A", "B"] for d in rebalances}, end)
    with pytest.raises(ValueError, match="no membership"):
        book_holdings(close, rebalances, lambda d: None)
