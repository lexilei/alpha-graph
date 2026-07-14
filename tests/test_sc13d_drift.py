"""Tests for C23 `sc13d_drift` (synthetic inputs only).

Pinned semantics under test (registration 2026-07-14, commit 77ffe85):
  - availability boundary: acceptance strictly before 16:00 ET on a trading day
    resolves to that close; at/after 16:00 (15:59 vs 16:00) or on a non-trading
    day, to the next trading close;
  - entry is the NEXT close after availability;
  - CAR is the holding return close(entry) -> close(entry+k), hand-computable;
  - control (a) same-ticker window at entry-252td, skipped on a +/-63td
    collision with an ORIGINAL 13D/13G event (amendments do not collide);
  - control (b) greedy nearest-acceptance original 13G on a DIFFERENT subject,
    no filing reused;
  - the pinned bar: BOTH paired HAC t >= +2.0 -> candidate, else rejected.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.signals.sc13d_drift import (
    add_control_a,
    add_control_b,
    availability_pos,
    build_events,
    car_curve,
    collides,
    collision_positions,
    curve_look,
    decide,
    eligible_13g,
    entry_pos,
    filing_grid_pos,
    build_real_events,
    paired_look,
)

CAL_START = "2015-01-01"
CAL_DAYS = 800


def _cal(start=CAL_START, days=CAL_DAYS):
    return pd.bdate_range(start, periods=days)


def _market(closes: dict, start=CAL_START, days=CAL_DAYS):
    """closes: {ticker: array-of-len-days or None for a flat finite series}."""
    cal = pd.bdate_range(start, periods=days)
    frames = []
    for tk, arr in closes.items():
        a = np.full(days, 100.0) if arr is None else np.asarray(arr, float)
        frames.append(pd.DataFrame({"date": cal, "ticker": tk, "close": a}))
    return pd.concat(frames, ignore_index=True), cal


def _mat(market):
    cm = market.pivot(index="date", columns="ticker", values="close").sort_index()
    return cm.to_numpy(), {tk: j for j, tk in enumerate(cm.columns)}


def _sc13(rows):
    base = dict(subject_cik="0000000000", accession=None, form="SC 13D",
                is_amendment=False, filing_date=None,
                acceptance_source="sgml_header", filer_name="X", filer_cik="1")
    out = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d["acceptance_ts"] = pd.Timestamp(d["acceptance_ts"])
        if d["accession"] is None:
            d["accession"] = f"acc{i}"
        if d["filing_date"] is None:
            d["filing_date"] = d["acceptance_ts"].normalize()
        out.append(d)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Availability boundary 15:59 / 16:00 and the next-close entry
# --------------------------------------------------------------------------- #

def test_availability_boundary_1559_same_close():
    cal = _cal()
    wed = pd.Timestamp("2016-06-01")               # a trading day in the panel
    assert wed in cal
    ap = availability_pos(pd.Timestamp("2016-06-01 15:59:00"), cal)
    assert ap == cal.get_loc(wed)                  # same-day close


def test_availability_boundary_1600_next_close():
    cal = _cal()
    ap = availability_pos(pd.Timestamp("2016-06-01 16:00:00"), cal)
    assert ap == cal.get_loc(pd.Timestamp("2016-06-02"))   # next trading close


def test_availability_after_hours_next_close():
    cal = _cal()
    ap = availability_pos(pd.Timestamp("2016-06-01 20:30:00"), cal)
    assert ap == cal.get_loc(pd.Timestamp("2016-06-02"))


def test_availability_non_trading_day_next_close():
    cal = _cal()
    sat = pd.Timestamp("2016-06-04")               # Saturday, not a trading day
    assert sat not in cal
    # even before 16:00, a non-trading day resolves to the next trading close
    ap = availability_pos(pd.Timestamp("2016-06-04 10:00:00"), cal)
    assert ap == cal.get_loc(pd.Timestamp("2016-06-06"))   # Monday


def test_next_close_entry_is_availability_plus_one():
    cal = _cal()
    ap = availability_pos(pd.Timestamp("2016-06-01 15:59:00"), cal)
    ep = entry_pos(pd.Timestamp("2016-06-01 15:59:00"), cal)
    assert ep == ap + 1
    assert cal[ep] == pd.Timestamp("2016-06-02")


def test_pre_calendar_acceptance_has_no_availability():
    cal = _cal()
    assert availability_pos(pd.Timestamp("2014-12-01 10:00:00"), cal) is None
    assert entry_pos(pd.Timestamp("2014-12-01 10:00:00"), cal) is None


# --------------------------------------------------------------------------- #
# CAR arithmetic, hand-computed
# --------------------------------------------------------------------------- #

def test_car_arithmetic_hand_computed():
    cal = _cal(days=400)
    close = np.full(len(cal), 50.0)                # finite everywhere
    e = 100
    close[e] = 100.0
    close[e + 1] = 101.0
    close[e + 5] = 105.0
    close[e + 21] = 110.0
    close[e + 42] = 120.0
    close[e + 63] = 130.0
    market, _ = _market({"AAA": close}, days=400)
    cm, tick_idx = _mat(market)
    cur = car_curve(cm, tick_idx, "AAA", e)
    assert cur[1] == pytest.approx(0.01)
    assert cur[5] == pytest.approx(0.05)
    assert cur[21] == pytest.approx(0.10)
    assert cur[42] == pytest.approx(0.20)
    assert cur[63] == pytest.approx(0.30)


def test_car_none_when_window_runs_past_calendar():
    cal = _cal(days=120)
    market, _ = _market({"AAA": None}, days=120)
    cm, tick_idx = _mat(market)
    assert car_curve(cm, tick_idx, "AAA", 100) is None   # 100+63 > 119
    assert car_curve(cm, tick_idx, "MISSING", 10) is None


def test_unpriced_subject_dropped():
    # AAA listed only from index 200 on (NaN before) -> a window at entry 50
    # is unpriced and dropped
    cal = _cal(days=500)
    close = np.full(len(cal), 100.0)
    close[:200] = np.nan
    market, _ = _market({"AAA": close}, days=500)
    cm, tick_idx = _mat(market)
    assert car_curve(cm, tick_idx, "AAA", 50) is None
    assert car_curve(cm, tick_idx, "AAA", 300) is not None


# --------------------------------------------------------------------------- #
# Real event assembly + drop reasons
# --------------------------------------------------------------------------- #

def test_build_real_events_priced_and_pre_calendar_drop():
    cal = _cal()
    market, _ = _market({"AAA": None})
    cm, tick_idx = _mat(market)
    sc13 = _sc13([
        {"subject_ticker": "AAA", "acceptance_ts": "2016-06-01 10:00"},
        {"subject_ticker": "AAA", "acceptance_ts": "2014-01-01 10:00"},  # pre
    ])
    ev = build_real_events(sc13, cal, cm, tick_idx)
    assert len(ev) == 2
    priced = ev[ev["real_priced"]]
    assert len(priced) == 1
    assert priced.iloc[0]["subject_ticker"] == "AAA"
    assert priced.iloc[0]["entry_date"] == pd.Timestamp("2016-06-02")
    dropped = ev[~ev["real_priced"]]
    assert dropped.iloc[0]["drop_reason"] == "pre_calendar"


def test_amendments_are_not_events():
    cal = _cal()
    market, _ = _market({"AAA": None})
    cm, tick_idx = _mat(market)
    sc13 = _sc13([
        {"subject_ticker": "AAA", "acceptance_ts": "2016-06-01 10:00"},
        {"subject_ticker": "AAA", "acceptance_ts": "2016-06-02 10:00",
         "form": "SC 13D/A", "is_amendment": True},
    ])
    ev = build_real_events(sc13, cal, cm, tick_idx)
    assert len(ev) == 1                             # the amendment is excluded


# --------------------------------------------------------------------------- #
# Control (a): -252td shift + collision skip (originals only)
# --------------------------------------------------------------------------- #

def _one_13d(acc="2016-06-01 10:00", ticker="AAA"):
    return {"subject_ticker": ticker, "acceptance_ts": acc}


def test_control_a_matches_when_clean():
    cal = _cal()
    market, _ = _market({"AAA": None})
    cm, tick_idx = _mat(market)
    sc13 = _sc13([_one_13d()])
    ev = build_real_events(sc13, cal, cm, tick_idx)
    ev = add_control_a(ev, sc13, cal, cm, tick_idx)
    row = ev.iloc[0]
    assert bool(row["real_priced"])
    assert bool(row["a_matched"])
    # control entry is exactly 252 td before the real entry
    assert not np.isnan(row["a_car63"])


def test_control_a_collision_with_original_13g_skips():
    cal = _cal()
    market, _ = _market({"AAA": None})
    cm, tick_idx = _mat(market)
    d13 = _one_13d()
    ep = entry_pos(pd.Timestamp("2016-06-01 10:00"), cal)
    cref = ep - 252
    coll_date = cal[cref]                            # land a 13G at the control anchor
    sc13 = _sc13([
        d13,
        {"subject_ticker": "AAA", "acceptance_ts": f"{coll_date.date()} 10:00",
         "form": "SC 13G", "is_amendment": False},
    ])
    ev = build_real_events(sc13, cal, cm, tick_idx)
    ev = add_control_a(ev, sc13, cal, cm, tick_idx)
    real = ev[ev["real_priced"]].iloc[0]
    assert not bool(real["a_matched"])
    assert real["a_skip_reason"] == "collision"


def test_control_a_amendment_does_not_collide():
    cal = _cal()
    market, _ = _market({"AAA": None})
    cm, tick_idx = _mat(market)
    ep = entry_pos(pd.Timestamp("2016-06-01 10:00"), cal)
    coll_date = cal[ep - 252]
    sc13 = _sc13([
        _one_13d(),
        {"subject_ticker": "AAA", "acceptance_ts": f"{coll_date.date()} 10:00",
         "form": "SC 13G/A", "is_amendment": True},     # amendment: not an event
    ])
    ev = build_real_events(sc13, cal, cm, tick_idx)
    ev = add_control_a(ev, sc13, cal, cm, tick_idx)
    real = ev[ev["real_priced"]].iloc[0]
    assert bool(real["a_matched"])


def test_collision_positions_originals_only():
    cal = _cal()
    sc13 = _sc13([
        {"subject_ticker": "AAA", "acceptance_ts": "2016-06-01 10:00",
         "form": "SC 13D", "is_amendment": False},
        {"subject_ticker": "AAA", "acceptance_ts": "2016-07-01 10:00",
         "form": "SC 13G", "is_amendment": False},
        {"subject_ticker": "AAA", "acceptance_ts": "2016-08-01 10:00",
         "form": "SC 13G/A", "is_amendment": True},        # excluded
    ])
    pos = collision_positions(sc13, cal)
    assert len(pos["AAA"]) == 2
    assert filing_grid_pos("2016-08-01 10:00", cal) not in set(pos["AAA"])


def test_collides_tolerance_inclusive():
    positions = np.array([100])
    assert collides(163, positions, tol=63) is True     # exactly 63 away
    assert collides(164, positions, tol=63) is False


# --------------------------------------------------------------------------- #
# Control (b): greedy nearest-first, different subject, no reuse
# --------------------------------------------------------------------------- #

def test_control_b_prefers_different_subject_and_nearest():
    cal = _cal()
    market, _ = _market({"AAA": None, "CCC": None})
    cm, tick_idx = _mat(market)
    # a 13G on AAA (same subject, nearest) must be rejected in favour of CCC
    sc13 = _sc13([
        {"subject_ticker": "AAA", "acceptance_ts": "2016-06-01 10:00",
         "form": "SC 13D", "is_amendment": False},
        {"subject_ticker": "AAA", "acceptance_ts": "2016-06-01 11:00",
         "form": "SC 13G", "is_amendment": False},          # same subject
        {"subject_ticker": "CCC", "acceptance_ts": "2016-06-05 10:00",
         "form": "SC 13G", "is_amendment": False},          # different subject
    ])
    ev = build_real_events(sc13, cal, cm, tick_idx)
    g13 = eligible_13g(sc13, cal, cm, tick_idx)
    ev = add_control_b(ev, g13)
    real = ev[ev["real_priced"]].iloc[0]
    assert bool(real["b_matched"])
    assert real["b_ticker"] == "CCC"


def test_control_b_greedy_no_reuse():
    cal = _cal()
    market, _ = _market({"AAA": None, "BBB": None, "CCC": None, "DDD": None})
    cm, tick_idx = _mat(market)
    # 13G_C (CCC) is the nearest to BOTH 13Ds; the closer 13D takes it, the
    # other cannot reuse it and falls back to 13G_D (DDD).
    sc13 = _sc13([
        {"subject_ticker": "AAA", "acceptance_ts": "2016-06-01 00:00",
         "form": "SC 13D", "is_amendment": False},
        {"subject_ticker": "BBB", "acceptance_ts": "2016-06-01 12:00",
         "form": "SC 13D", "is_amendment": False},
        {"subject_ticker": "CCC", "acceptance_ts": "2016-06-01 05:00",
         "form": "SC 13G", "is_amendment": False},   # 5h from AAA, 7h from BBB
        {"subject_ticker": "DDD", "acceptance_ts": "2016-06-20 00:00",
         "form": "SC 13G", "is_amendment": False},   # far, only a fallback
    ])
    ev = build_real_events(sc13, cal, cm, tick_idx)
    g13 = eligible_13g(sc13, cal, cm, tick_idx)
    ev = add_control_b(ev, g13)
    by_tk = ev.set_index("subject_ticker")
    assert by_tk.loc["AAA", "b_ticker"] == "CCC"     # nearest wins its first pick
    assert by_tk.loc["BBB", "b_ticker"] == "DDD"     # no reuse -> fallback
    # every assigned 13G is distinct
    used = ev.loc[ev["b_matched"], "b_accession"]
    assert used.is_unique


# --------------------------------------------------------------------------- #
# Paired looks, curve consistency, and the pinned decision bar
# --------------------------------------------------------------------------- #

def test_curve_63_equals_paired_a_mean_diff():
    cal = _cal()
    rng = np.random.default_rng(0)
    tickers = [f"T{i:02d}" for i in range(8)]
    closes = {tk: 100.0 * np.cumprod(1 + rng.normal(0, 0.01, CAL_DAYS))
              for tk in tickers}
    market, _ = _market(closes)
    rows = []
    for i, tk in enumerate(tickers):
        d = pd.Timestamp("2016-06-01") + pd.Timedelta(days=3 * i)
        rows.append({"subject_ticker": tk, "acceptance_ts": f"{d.date()} 10:00",
                     "form": "SC 13D", "is_amendment": False})
    sc13 = _sc13(rows)
    ev = build_events(sc13, market)
    la = paired_look(ev, "a")
    curve = curve_look(ev)
    c63 = curve[curve["offset_td"] == 63]["mean_excess"].iloc[0]
    assert c63 == pytest.approx(la["mean_diff"])


def test_decide_bar_requires_both():
    assert decide({"t_hac": 2.5}, {"t_hac": 2.1}) == "candidate"
    assert decide({"t_hac": 2.5}, {"t_hac": 1.9}) == "rejected"
    assert decide({"t_hac": 1.5}, {"t_hac": 3.0}) == "rejected"
    assert decide({"t_hac": 2.0}, {"t_hac": 2.0}) == "candidate"   # inclusive


def test_build_events_full_schema_and_counts():
    cal = _cal()
    market, _ = _market({"AAA": None, "CCC": None})
    sc13 = _sc13([
        {"subject_ticker": "AAA", "acceptance_ts": "2016-06-01 10:00",
         "form": "SC 13D", "is_amendment": False},
        {"subject_ticker": "CCC", "acceptance_ts": "2016-06-05 10:00",
         "form": "SC 13G", "is_amendment": False},
    ])
    ev = build_events(sc13, market)
    for col in ("subject_ticker", "acceptance_ts", "avail_date", "entry_date",
                "car63", "a_car63", "b_car63", "a_matched", "b_matched",
                "real_priced", "drop_reason"):
        assert col in ev.columns
    assert int(ev["real_priced"].sum()) == 1        # only the 13D is an event
