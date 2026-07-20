"""C34 short_interest_xs: symbol mapping, PIT denominator (filed <= avail —
the C33-audit lesson), exclusions."""
import pandas as pd
import pytest

from alpha_graph.signals import short_interest_xs as six


def si_frame(rows):
    return pd.DataFrame(
        rows, columns=["symbol", "avail_date", "short_qty"]
    ).assign(adv_qty=1.0, days_to_cover=1.0, market_class="NNM")


def test_pit_denominator_filed_before_avail(monkeypatch):
    """The denominator must be the count FILED on or before avail — a
    count filed after avail (even with an earlier 'end') must not leak."""
    calls = {}

    def fake_shares_asof(ticker, dates):
        calls[ticker] = list(pd.to_datetime(dates))
        # 100 shares known as of any date passed in
        return pd.Series(100.0, index=pd.DatetimeIndex(pd.to_datetime(dates)))

    monkeypatch.setattr(six, "shares_asof", fake_shares_asof)
    si = si_frame([("AAPL", "2024-01-10", 500.0)])
    res = six.build_short_interest_xs(si, {"AAPL"})
    # shares_asof consulted AT the avail date — PIT by construction
    assert calls["AAPL"] == [pd.Timestamp("2024-01-10")]
    assert res["short_interest_xs"].iloc[0] == pytest.approx(5.0)


def test_symbol_map_and_exclusions(monkeypatch):
    monkeypatch.setattr(
        six, "shares_asof",
        lambda t, d: pd.Series(10.0, index=pd.DatetimeIndex(pd.to_datetime(d))),
    )
    si = si_frame(
        [("BFB", "2024-01-10", 20.0), ("BRKB", "2024-01-10", 30.0),
         ("ZZZC", "2024-01-10", 40.0)]
    )
    res = six.build_short_interest_xs(si, {"BF-B", "BRK-B", "AAPL"})
    assert set(res["ticker"]) == {"BF-B"}   # BRKB excluded, ZZZC off-panel
    assert res["short_interest_xs"].iloc[0] == pytest.approx(2.0)


def test_no_share_facts_dropped(monkeypatch):
    def raising(t, d):
        raise KeyError("no facts")
    monkeypatch.setattr(six, "shares_asof", raising)
    si = si_frame([("AAPL", "2024-01-10", 500.0)])
    res = six.build_short_interest_xs(si, {"AAPL"})
    assert len(res) == 0
