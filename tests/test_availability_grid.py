"""Tests for plan Items 2+4: t+1 availability at the merge layer and the
daily as-of signal grid (C15 daily 8-K frequency, C10 daily customer
momentum).

Conventions under test:
  - builders stamp rows with the COMPUTATION date t (data through close t);
  - the merge layer is the single owner of the availability lag, in calendar
    days, so lag=1 means "first usable at the next trading close"
    (Friday -> Monday across the weekend);
  - therefore builder + lag=1 merge composes to exactly one day of lag
    (the no-double-lag gate).
"""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.signals import event_freq_8k as ef
from alpha_graph.signals.graph_signal import (
    compute_customer_momentum_daily,
    compute_customer_momentum_timeseries,
)
from alpha_graph.signals.ml_combiner import _merge_asof_signal


# --------------------------------------------------------------------------- #
# Merge-layer availability lag
# --------------------------------------------------------------------------- #

def _tiny_panel():
    # Thu, Fri, Mon, Tue around a weekend
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    return pd.DataFrame({"ticker": "X", "date": dates})


def test_merge_lag_zero_same_day_attaches():
    sig = pd.DataFrame({"ticker": ["X"], "filing_date": [pd.Timestamp("2020-01-03")],
                        "v": [1.0]})
    m = _merge_asof_signal(_tiny_panel(), sig, "date", "filing_date", "ticker", ["v"])
    got = m.set_index("date")["v"]
    assert np.isnan(got[pd.Timestamp("2020-01-02")])
    assert got[pd.Timestamp("2020-01-03")] == 1.0      # same-close attach (lag 0)


def test_merge_lag_one_excludes_same_day():
    sig = pd.DataFrame({"ticker": ["X"], "filing_date": [pd.Timestamp("2020-01-06")],
                        "v": [1.0]})
    m = _merge_asof_signal(_tiny_panel(), sig, "date", "filing_date", "ticker", ["v"],
                           availability_lag_days=1)
    got = m.set_index("date")["v"]
    assert np.isnan(got[pd.Timestamp("2020-01-06")])   # Monday filing not usable Monday
    assert got[pd.Timestamp("2020-01-07")] == 1.0      # first use Tuesday


def test_merge_lag_one_friday_to_monday():
    # Friday filing shifted to Saturday attaches at the NEXT trading close
    sig = pd.DataFrame({"ticker": ["X"], "filing_date": [pd.Timestamp("2020-01-03")],
                        "v": [2.5]})
    m = _merge_asof_signal(_tiny_panel(), sig, "date", "filing_date", "ticker", ["v"],
                           availability_lag_days=1)
    got = m.set_index("date")["v"]
    assert np.isnan(got[pd.Timestamp("2020-01-03")])
    assert got[pd.Timestamp("2020-01-06")] == 2.5


def test_lag_controls_shift_semantics():
    # build_feature_panel(lag_controls=True) lags the in-panel controls with
    # the IDENTICAL expression tested here: panel sorted by (ticker, date),
    # then panel.groupby("ticker")[cols].shift(1). Verify one-trading-day
    # semantics within ticker and NaN on each ticker's first row.
    dates = pd.bdate_range("2020-01-06", periods=5)
    cols = ["momentum_21d", "volume_zscore"]
    df = pd.concat([
        pd.DataFrame({"ticker": t, "date": dates,
                      "momentum_21d": base + np.arange(5.0),
                      "volume_zscore": base - np.arange(5.0)})
        for t, base in [("A", 10.0), ("B", 100.0)]
    ], ignore_index=True).sort_values(["ticker", "date"])
    unshifted = df[cols].copy()
    df[cols] = df.groupby("ticker")[cols].shift(1)
    for t in ["A", "B"]:
        got = df[df["ticker"] == t].reset_index(drop=True)
        raw = unshifted[(df["ticker"] == t).values].reset_index(drop=True)
        for c in cols:
            assert np.isnan(got.loc[0, c])                       # first row NaN
            assert got.loc[1:, c].tolist() == raw.loc[:3, c].tolist()  # col(t)==raw(t-1)


# --------------------------------------------------------------------------- #
# C15 daily 8-K frequency builder
# --------------------------------------------------------------------------- #

def _cal(n=30, start="2021-01-04"):
    return pd.DatetimeIndex(pd.bdate_range(start, periods=n))


def _events(ticker, days, cal):
    return pd.DataFrame({"ticker": ticker, "fdate": [cal[d] for d in days]})


def test_evt8k_daily_hand_computed_z(monkeypatch):
    # WINDOW=3, BASELINE=10, MIN_OBS=5: fully hand-checkable.
    monkeypatch.setattr(ef, "DAILY_WINDOW", 3)
    monkeypatch.setattr(ef, "DAILY_BASELINE", 10)
    monkeypatch.setattr(ef, "DAILY_MIN_OBS", 5)
    cal = _cal(30)
    # one filing every 3rd day through d12, then quiet, then a 3-day burst
    ev = _events("A", [0, 3, 6, 9, 12, 20, 21, 22], cal)
    df = ef.compute_daily(start=str(cal[0].date()), events=ev, calendar=cal)
    z = df.set_index("date")["evt8k_freq_z_d"]
    # at d22: count_3 = 3 (burst); baseline obs = c3(d10..d19) = five 1s, five 0s
    # -> mu 0.5, sd sqrt(5/18); z = 2.5 / sqrt(5/18)
    expect = 2.5 / np.sqrt(5.0 / 18.0)
    assert z[cal[22]] == pytest.approx(expect, rel=1e-12)


def test_evt8k_daily_counts_own_day_and_is_pit():
    cal = _cal(220)
    ev = _events("A", list(range(0, 200, 2)), cal)   # steady filer
    base = ef.compute_daily(start=str(cal[0].date()), events=ev, calendar=cal)
    c = base.set_index("date")["evt8k_count_21d"]
    # builder stamps the computation date: a filing at t is inside count(t)
    t = cal[198]                                      # a filing day (even index)
    assert c[t] >= 1.0
    # PIT: appending a FUTURE filing must not change any value at or before t
    ev2 = pd.concat([ev, _events("A", [219], cal)], ignore_index=True)
    pert = ef.compute_daily(start=str(cal[0].date()), events=ev2, calendar=cal)
    a = base[base["date"] <= t].reset_index(drop=True)
    b = pert[pert["date"] <= t].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_evt8k_daily_weekend_filing_maps_to_next_close(monkeypatch):
    monkeypatch.setattr(ef, "DAILY_WINDOW", 1)
    monkeypatch.setattr(ef, "DAILY_BASELINE", 10)
    monkeypatch.setattr(ef, "DAILY_MIN_OBS", 2)
    cal = _cal(20)                                    # bdays only
    sat = cal[9] + pd.Timedelta(days=1)               # Friday + 1 = Saturday
    assert sat.dayofweek == 5
    ev = pd.DataFrame({"ticker": ["A", "A"], "fdate": [cal[0], sat]})
    df = ef.compute_daily(start=str(cal[0].date()), events=ev, calendar=cal)
    c = df.set_index("date")["evt8k_count_21d"]
    assert c[cal[10]] == 1.0                          # Monday close counts it
    # and z is defined there (nonconstant baseline not needed for the count)


def test_evt8k_daily_constant_baseline_is_nan(monkeypatch):
    monkeypatch.setattr(ef, "DAILY_WINDOW", 3)
    monkeypatch.setattr(ef, "DAILY_BASELINE", 10)
    monkeypatch.setattr(ef, "DAILY_MIN_OBS", 5)
    cal = _cal(30)
    ev = _events("A", [0, 25], cal)                   # long all-zero baseline
    df = ef.compute_daily(start=str(cal[0].date()), events=ev, calendar=cal)
    # d25 burst sits on an all-zero baseline window -> sd 0 -> NaN (dropped);
    # rows that survive must carry finite z only
    assert (df["date"] != cal[25]).all() or np.isfinite(
        df.set_index("date")["evt8k_freq_z_d"].get(cal[25], np.nan))
    assert df["evt8k_freq_z_d"].notna().all()


def test_evt8k_daily_min_obs_gate(monkeypatch):
    monkeypatch.setattr(ef, "DAILY_WINDOW", 3)
    monkeypatch.setattr(ef, "DAILY_BASELINE", 10)
    monkeypatch.setattr(ef, "DAILY_MIN_OBS", 5)
    cal = _cal(30)
    ev = _events("A", [0, 3, 6, 9, 12, 20, 21, 22], cal)
    df = ef.compute_daily(start=str(cal[0].date()), events=ev, calendar=cal)
    # base = c3.shift(3), c3 defined from index 2 -> base from index 5;
    # MIN_OBS=5 -> first possible z at index 9
    assert df["date"].min() >= cal[9]


def test_no_double_lag_composition(monkeypatch):
    # builder(t) + merge lag=1 == previous trading day's builder value
    monkeypatch.setattr(ef, "DAILY_WINDOW", 3)
    monkeypatch.setattr(ef, "DAILY_BASELINE", 10)
    monkeypatch.setattr(ef, "DAILY_MIN_OBS", 5)
    cal = _cal(40)
    rng = np.random.default_rng(7)
    days = sorted(rng.choice(np.arange(38), size=14, replace=False).tolist())
    ev = _events("A", days, cal)
    sig = ef.compute_daily(start=str(cal[0].date()), events=ev, calendar=cal)
    panel = pd.DataFrame({"ticker": "A", "date": cal})
    merged = _merge_asof_signal(panel, sig.rename(columns={"date": "sdate"}),
                                "date", "sdate", "ticker", ["evt8k_freq_z_d"],
                                availability_lag_days=1)
    got = merged.set_index("date")["evt8k_freq_z_d"]
    raw = sig.set_index("date")["evt8k_freq_z_d"].reindex(cal)
    # every day with a defined previous-day z must see exactly that value
    prev = raw.shift(1)
    both = prev.notna() & got.notna()
    assert both.sum() > 10
    assert np.allclose(got[both], prev[both], equal_nan=False)
    # and never the SAME day's value when it differs from yesterday's
    diff = prev.notna() & raw.notna() & (prev != raw)
    assert diff.any()
    assert not np.allclose(got[diff], raw[diff])


# --------------------------------------------------------------------------- #
# C10 daily customer momentum
# --------------------------------------------------------------------------- #

def _cm_inputs():
    cal = pd.bdate_range("2020-01-01", "2020-03-31")
    rows = []
    for i, d in enumerate(cal):
        rows.append({"ticker": "A", "date": d, "close": 50.0 + 0.1 * i})
        rows.append({"ticker": "B", "date": d, "close": 100.0 * (1.01 ** i)})
    market = pd.DataFrame(rows)
    rel = pd.DataFrame([{
        "source": "A", "target": "B", "relation": "customer",
        "confidence": 0.9, "filing_date": cal[10], "evidence": "synthetic",
    }])
    return cal, market, rel


def test_cust_mom_daily_exact_value_and_edge_availability():
    cal, market, rel = _cm_inputs()
    df = compute_customer_momentum_daily(
        start_date=str(cal[0].date()), end_date=str(cal[-1].date()),
        min_conf=0.8, horizon=2, relationships=rel, market=market,
    )
    # no rows before the edge's filing date (graph empty -> day skipped)
    assert df["date"].min() == cal[10]
    a = df[df["ticker"] == "A"].set_index("date")["spillover_cust_mom"]
    # A's single customer B: value = B's 2-day momentum = 1.01^2 - 1
    expect = round(1.01 ** 2 - 1.0, 6)
    assert a[cal[10]] == pytest.approx(expect, abs=1e-6)
    assert a[cal[-1]] == pytest.approx(expect, abs=1e-6)
    # B has no customers -> NaN
    b = df[df["ticker"] == "B"].set_index("date")["spillover_cust_mom"]
    assert np.isnan(b[cal[10]])


def test_cust_mom_daily_matches_monthly_at_month_ends():
    # Grid is a convention: at each monthly grid date, the daily builder's
    # value at the last trading day <= that date must equal the monthly value.
    cal, market, rel = _cm_inputs()
    daily = compute_customer_momentum_daily(
        start_date=str(cal[0].date()), end_date=str(cal[-1].date()),
        min_conf=0.8, horizon=2, relationships=rel, market=market,
    )
    monthly = compute_customer_momentum_timeseries(
        start_date=str(cal[0].date()), end_date=str(cal[-1].date()),
        min_conf=0.8, horizon=2, relationships=rel, market=market,
    )
    d_a = daily[daily["ticker"] == "A"].set_index("date")["spillover_cust_mom"]
    m_a = monthly[monthly["ticker"] == "A"].set_index("date")["spillover_cust_mom"]
    assert len(m_a) == 3                                   # Jan, Feb, Mar ends
    for me, mv in m_a.items():
        last_td = d_a.index[d_a.index <= me].max()
        assert d_a[last_td] == pytest.approx(mv, abs=1e-6), me
