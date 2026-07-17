"""C30 volume_trend builder: slope recovery, min-month gate, partial-month
exclusion on synthetic panels."""
import numpy as np
import pandas as pd

from alpha_graph.signals.volume_trend import (
    MIN_MONTHS,
    _monthly_volume,
    build_volume_trend,
)


def daily_panel(ticker, monthly_vols, start="2015-01-01"):
    """Daily rows spreading each month's volume over its business days."""
    rows = []
    month = pd.Period(start, freq="M")
    for mv in monthly_vols:
        days = pd.bdate_range(month.start_time, month.end_time)
        for d in days:
            rows.append({"ticker": ticker, "date": d, "volume": mv / len(days)})
        month += 1
    return pd.DataFrame(rows)


def test_recovers_linear_trend():
    vols = [1000.0 + 10.0 * m for m in range(70)]
    res = build_volume_trend(daily_panel("A", vols))
    assert not res.empty
    last = res.iloc[-1]
    # trailing 60-month window: months 10..69, mean = 1000 + 10*39.5
    expect = 10.0 / (1000.0 + 10.0 * 39.5)
    assert np.isclose(last["volume_trend"], expect, rtol=1e-6)
    # flat volume -> zero trend
    flat = build_volume_trend(daily_panel("A", [500.0] * 40))
    assert np.allclose(flat["volume_trend"], 0.0)


def test_min_months_gate():
    res = build_volume_trend(daily_panel("A", [1000.0] * (MIN_MONTHS - 1)))
    assert res.empty
    res = build_volume_trend(daily_panel("A", [1000.0] * MIN_MONTHS))
    assert len(res) == 1  # exactly the 30th month qualifies


def test_partial_month_excluded():
    full = daily_panel("A", [1000.0] * 3)
    # B trades only the last 5 days of month 1, then fully
    b = daily_panel("B", [1000.0] * 3)
    first_month = b["date"].dt.to_period("M") == b["date"].dt.to_period("M").min()
    keep = b[~first_month | (b["date"] >= b[first_month]["date"].max() - pd.Timedelta(days=6))]
    m = _monthly_volume(pd.concat([full, keep], ignore_index=True))
    b_months = m[m["ticker"] == "B"]
    assert len(b_months) == 2  # partial first month dropped
    a_months = m[m["ticker"] == "A"]
    assert len(a_months) == 3  # full months all kept
