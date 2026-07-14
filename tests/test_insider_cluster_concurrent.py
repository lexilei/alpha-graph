"""Tests for C20 `insider_cluster_concurrent` (synthetic inputs only).

Pinned semantics under test (registration 2026-07-14 + the warmup
build-time pin):
  - concurrency count window: (d - 21 calendar days, d] — an event exactly
    21 days earlier does NOT count, 20 days does; same-day events and the
    event itself count;
  - PIT percentile: an event's qualification (and every concurrency stat)
    must not change when FUTURE events are added;
  - warmup: fewer than `min_prior` (production 30) strictly-prior events in
    the trailing 3-year window (both interval ends open, DateOffset years)
    -> the event does not qualify; the 80th-percentile threshold is
    np.percentile linear interpolation and ties qualify (>=);
  - construction identity: the C20 series is `insider_cluster.build_series`
    verbatim on the qualifying subset — same events in -> same series out
    when all qualify;
  - non-qualifying events contribute nothing to the series.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.signals.insider_cluster import build_series
from alpha_graph.signals.insider_cluster_concurrent import (
    build_c20,
    concurrency_stats,
)


def _ev(rows):
    return pd.DataFrame(
        [{"ticker": tk, "event_date": pd.Timestamp(d)} for tk, d in rows])


def _market(tickers=("AAA",), start="2020-01-01", days=220, ret=0.01,
            volume=1_000_000.0):
    """Business-day panel with an exact constant daily return per ticker."""
    cal = pd.bdate_range(start, periods=days)
    frames = []
    for tk in tickers:
        close = 100.0 * np.cumprod(np.full(days, 1.0 + ret))
        frames.append(pd.DataFrame(
            {"date": cal, "ticker": tk, "close": close, "volume": volume}))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 21-calendar-day count window boundary
# --------------------------------------------------------------------------- #

def test_count_window_half_open_21_day_boundary():
    stats = concurrency_stats(_ev([
        ("T1", "2020-02-03"), ("T2", "2020-02-24"),   # gap exactly 21 days
        ("T3", "2020-05-10"), ("T4", "2020-05-30"),   # gap 20 days
    ]), min_prior=0)
    by_date = stats.set_index("event_date")["conc_count"]
    assert by_date[pd.Timestamp("2020-02-03")] == 1   # isolated: self only
    assert by_date[pd.Timestamp("2020-02-24")] == 1   # 21 days out: excluded
    assert by_date[pd.Timestamp("2020-05-30")] == 2   # 20 days out: counted


def test_count_window_includes_same_day_events_and_self():
    stats = concurrency_stats(_ev([
        ("T1", "2020-08-10"), ("T2", "2020-08-10"),
    ]), min_prior=0)
    assert (stats["conc_count"] == 2).all()


# --------------------------------------------------------------------------- #
# PIT: future events change nothing earlier
# --------------------------------------------------------------------------- #

def test_future_events_do_not_change_earlier_stats_or_qualification():
    base = _ev([("T1", "2020-01-10"), ("T2", "2020-01-15"),
                ("T3", "2020-02-01"), ("T4", "2020-03-01")])
    future = _ev([("T5", "2020-03-02"), ("T6", "2020-06-01")])
    s_base = concurrency_stats(base, min_prior=2)
    s_both = concurrency_stats(pd.concat([base, future], ignore_index=True),
                               min_prior=2)
    # the crafted base holds a mixed pattern so the equality is meaningful
    assert list(s_base["qualifies"]) == [False, False, True, False]
    pd.testing.assert_frame_equal(s_both.iloc[:len(base)], s_base)


# --------------------------------------------------------------------------- #
# Warmup rule + the open 3-year boundary
# --------------------------------------------------------------------------- #

def _warmup_events(boundary_date):
    """`boundary_date` + 29 isolated events strictly inside the target's
    3y window + the isolated target at 2023-06-15 (its count is 1)."""
    rows = [("B0", boundary_date)]
    d = pd.Timestamp("2020-08-01")
    for k in range(29):
        rows.append((f"P{k:02d}", d + pd.Timedelta(days=30 * k)))
    rows.append(("TGT", "2023-06-15"))
    return _ev(rows)


def test_fewer_than_30_prior_events_in_3y_never_qualifies():
    # the boundary event sits EXACTLY 3 years before the target: the open
    # interval (d - 3y, d) excludes it -> 29 prior events -> warmup blocks
    stats = concurrency_stats(_warmup_events("2020-06-15"))
    tgt = stats.iloc[-1]
    assert tgt["n_prior_3y"] == 29
    assert not tgt["qualifies"]
    assert np.isnan(tgt["pctl_thresh"])


def test_exactly_30_prior_events_can_qualify_and_ties_pass():
    # one day inside the 3y window -> 30 prior events, all with count 1:
    # threshold = percentile_80({1,...,1}) = 1.0; the target's count 1
    # qualifies on the tie (>=)
    stats = concurrency_stats(_warmup_events("2020-06-16"))
    tgt = stats.iloc[-1]
    assert tgt["n_prior_3y"] == 30
    assert tgt["conc_count"] == 1
    assert tgt["pctl_thresh"] == pytest.approx(1.0)
    assert tgt["qualifies"]


# --------------------------------------------------------------------------- #
# Percentile threshold: exact value, PIT distribution, below-threshold fails
# --------------------------------------------------------------------------- #

def test_percentile_threshold_exact_and_binding():
    # priors: 4 isolated (count 1) + a same-day pair (counts 2, 2)
    rows = [("T1", "2020-01-05"), ("T2", "2020-02-05"),
            ("T3", "2020-03-05"), ("T4", "2020-04-05"),
            ("P1", "2020-05-05"), ("P2", "2020-05-05"),
            ("X", "2020-06-05"),                       # isolated: count 1
            ("Y", "2020-07-05"), ("Z", "2020-07-05")]  # pair: count 2
    stats = concurrency_stats(_ev(rows), min_prior=5).set_index("ticker")

    x = stats.loc["X"]
    assert x["conc_count"] == 1 and x["n_prior_3y"] == 6
    assert x["pctl_thresh"] == pytest.approx(
        np.percentile([1, 1, 1, 1, 2, 2], 80))          # = 2.0
    assert not x["qualifies"]                           # 1 < 2.0

    y = stats.loc["Y"]
    assert y["conc_count"] == 2 and y["n_prior_3y"] == 7
    assert y["pctl_thresh"] == pytest.approx(
        np.percentile([1, 1, 1, 1, 2, 2, 1], 80))       # = 1.8
    assert y["qualifies"]                               # 2 >= 1.8


# --------------------------------------------------------------------------- #
# Construction identity with the C16 machinery
# --------------------------------------------------------------------------- #

def test_series_identical_to_c16_machinery_when_all_qualify():
    # two tickers, one event every 5 calendar days: counts are monotone
    # non-decreasing, so with min_prior=0 every event qualifies (the first
    # vacuously); the C20 series must then equal build_series on the same
    # events byte-for-byte
    rows = [(("AAA", "BBB")[k % 2],
             pd.Timestamp("2020-02-03") + pd.Timedelta(days=5 * k))
            for k in range(20)]
    events = _ev(rows)
    market = _market(tickers=("AAA", "BBB"))

    series_c20, qual, stats = build_c20(events, market, min_prior=0)
    assert stats["qualifies"].all()
    assert len(qual) == len(events)

    series_c16, _ = build_series(events, market)
    pd.testing.assert_frame_equal(series_c20, series_c16)
    assert series_c16["n_active"].max() > 0             # non-degenerate case


def test_non_qualifying_events_contribute_nothing():
    # five isolated warmup-blocked events + one qualifying event (count 2,
    # 5 priors); only the qualifying event's 63-day window is active
    rows = [("T1", "2020-02-03"), ("T2", "2020-03-14"),
            ("T3", "2020-04-23"), ("T4", "2020-06-02"),
            ("T5", "2020-07-12"), ("T6", "2020-07-22")]
    market = _market(tickers=("T1", "T2", "T3", "T4", "T5", "T6"))
    series, qual, stats = build_c20(_ev(rows), market, min_prior=5)

    assert list(qual["ticker"]) == ["T6"]
    assert stats["qualifies"].sum() == 1
    assert int(series["n_active"].sum()) == 63          # one 63-day window
    assert int(series["n_active"].max()) == 1
    # nothing active before the qualifying event's entry
    assert (series.loc[:pd.Timestamp("2020-07-22"), "n_active"] == 0).all()
