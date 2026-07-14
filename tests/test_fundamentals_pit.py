"""Tests for the B7-B9 PIT fundamentals controls (synthetic inputs only).

Builder (signals/fundamentals_pit.py):
  - first-filed discipline: a later revision of the same (cik, ddate) never
    replaces the original book value;
  - NCI fallback fires only where the parent equity tag is absent for the
    filer-period;
  - TTM avail_date = max of the 4 constituent filings' filed dates;
  - Q4-derived net income (annual - 3 quarters) flows into the TTM with the
    ANNUAL filing's availability;
  - a missing quarter (window span > TTM_MAX_SPAN_DAYS) leaves TTM undefined;
  - negative book equity passes through (real);
  - PIT perturbation: appending later filings changes nothing dated earlier.

Panel wiring (ml_combiner.build_feature_panel):
  - B7 log_mktcap_pit = log(latest share count FILED on or before t x close),
    BRK-B excluded, NaN before the first share filing;
  - B8/B9 attach as-of avail_date with the v0 t+1 availability lag
    (Friday -> Monday across a weekend), divided by the market cap at t;
  - negative equity produces a negative book_to_market_pit;
  - missing caches leave all three columns NaN.
"""

import numpy as np
import pandas as pd
import pytest

import alpha_graph.signals.ml_combiner as ml
from alpha_graph.signals.fundamentals_pit import (
    TAG_EQUITY,
    TAG_EQUITY_NCI,
    TAG_NI,
    book_equity_events,
    compute,
    net_income_ttm_events,
)


def _fact(cik="0000000001", ticker="AAA", ddate="2020-03-31", qtrs=0, value=1.0,
          tag=TAG_EQUITY, uom="USD", filed="2020-05-01", accepted=None,
          adsh="a1", period=None, form="10-Q"):
    filed = pd.Timestamp(filed)
    return dict(
        cik=cik, ticker=ticker, ddate=pd.Timestamp(ddate), qtrs=qtrs,
        value=value, tag=tag, uom=uom, filed=filed,
        accepted=pd.Timestamp(accepted) if accepted else filed + pd.Timedelta(hours=12),
        adsh=adsh, period=pd.Timestamp(period) if period else pd.Timestamp(ddate),
        form=form,
    )


def _ni(ddate, value, filed, adsh, qtrs=1, form="10-Q", period=None):
    return _fact(ddate=ddate, qtrs=qtrs, value=value, tag=TAG_NI,
                 filed=filed, adsh=adsh, form=form, period=period)


# --------------------------------------------------------------------------- #
# Book equity: first-filed + NCI fallback + sign
# --------------------------------------------------------------------------- #

def test_book_first_filed_beats_revision():
    facts = pd.DataFrame([
        _fact(value=5e9, filed="2020-05-01", adsh="orig"),
        _fact(value=9e9, filed="2021-05-01", adsh="restated", period="2021-03-31"),
    ])
    ev = book_equity_events(facts)
    assert len(ev) == 1
    assert ev.iloc[0]["book_equity"] == 5e9
    assert ev.iloc[0]["avail_date"] == pd.Timestamp("2020-05-01")


def test_book_nci_fallback_only_when_parent_absent():
    facts = pd.DataFrame([
        # period 1: both tags -> parent wins even though NCI differs
        _fact(ddate="2020-03-31", value=5e9, adsh="p1", filed="2020-05-01"),
        _fact(ddate="2020-03-31", value=6e9, adsh="p1", filed="2020-05-01",
              tag=TAG_EQUITY_NCI),
        # period 2: parent absent -> NCI fallback
        _fact(ddate="2020-06-30", value=7e9, adsh="p2", filed="2020-08-01",
              tag=TAG_EQUITY_NCI),
    ])
    ev = book_equity_events(facts).set_index("period_end")["book_equity"]
    assert ev[pd.Timestamp("2020-03-31")] == 5e9
    assert ev[pd.Timestamp("2020-06-30")] == 7e9


def test_book_non_instant_rows_ignored():
    facts = pd.DataFrame([
        _fact(value=5e9, adsh="good"),
        _fact(ddate="2020-06-30", value=8e9, adsh="dur", qtrs=1,
              filed="2020-08-01"),
    ])
    ev = book_equity_events(facts)
    assert len(ev) == 1
    assert ev.iloc[0]["period_end"] == pd.Timestamp("2020-03-31")


def test_negative_book_equity_passes_through():
    facts = pd.DataFrame([_fact(value=-8.2e9)])
    df = compute(facts=facts)
    assert len(df) == 1
    assert df.iloc[0]["book_equity"] == -8.2e9


# --------------------------------------------------------------------------- #
# Net income TTM: availability, Q4 derivation, gap guard
# --------------------------------------------------------------------------- #

QENDS = [f"2020-{m:02d}-{d}" for m, d in [(3, 31), (6, 30), (9, 30), (12, 31)]]
FILED = ["2020-05-01", "2020-08-01", "2020-11-01", "2021-02-15"]


def test_ttm_avail_is_max_of_constituent_filed_dates():
    vals = [1e9, 2e9, 3e9, 4e9]
    facts = pd.DataFrame([
        _ni(d, v, f, adsh=f"q{i}")
        for i, (d, v, f) in enumerate(zip(QENDS, vals, FILED))
    ])
    ev = net_income_ttm_events(facts)
    assert len(ev) == 1
    r = ev.iloc[0]
    assert r["net_income_ttm"] == 10e9
    assert r["period_end"] == pd.Timestamp("2020-12-31")
    assert r["avail_date"] == pd.Timestamp("2021-02-15")   # max of the four


def test_ttm_avail_max_even_when_old_quarter_filed_last():
    # Q2 filed delinquently AFTER Q4: the TTM completes only then.
    filed = ["2020-05-01", "2021-03-20", "2020-11-01", "2021-02-15"]
    facts = pd.DataFrame([
        _ni(d, 1e9, f, adsh=f"q{i}")
        for i, (d, f) in enumerate(zip(QENDS, filed))
    ])
    ev = net_income_ttm_events(facts)
    assert ev.iloc[0]["avail_date"] == pd.Timestamp("2021-03-20")


def test_ttm_q4_derived_path_and_annual_availability():
    # Q1-Q3 standalone + annual 10-K only: Q4 = 10e9 - 6e9; TTM = annual.
    facts = pd.DataFrame([
        _ni("2020-03-31", 1e9, "2020-05-01", "q1"),
        _ni("2020-06-30", 2e9, "2020-08-01", "q2"),
        _ni("2020-09-30", 3e9, "2020-11-01", "q3"),
        _ni("2020-12-31", 10e9, "2021-02-15", "k", qtrs=4, form="10-K",
            period="2020-12-31"),
    ])
    ev = net_income_ttm_events(facts)
    assert len(ev) == 1
    r = ev.iloc[0]
    assert r["net_income_ttm"] == 10e9                     # 1+2+3+(10-6)
    assert r["avail_date"] == pd.Timestamp("2021-02-15")   # the ANNUAL filing


def test_ttm_undefined_across_missing_quarter():
    # Q2 2020 missing entirely: 2020-03-31..2021-03-31 spans 365d > 320d.
    facts = pd.DataFrame([
        _ni("2020-03-31", 1e9, "2020-05-01", "q1"),
        _ni("2020-09-30", 3e9, "2020-11-01", "q3"),
        _ni("2020-12-31", 4e9, "2021-02-15", "q4"),
        _ni("2021-03-31", 5e9, "2021-05-01", "q5"),
    ])
    ev = net_income_ttm_events(facts)
    assert ev.empty


# --------------------------------------------------------------------------- #
# PIT perturbation + state collapse
# --------------------------------------------------------------------------- #

def _quarter_history(n=8, start="2019-03-31"):
    qends = pd.date_range(start, periods=n, freq="QE")
    rows = []
    for i, d in enumerate(qends):
        f = d + pd.Timedelta(days=35)
        rows.append(_fact(ddate=d, value=(5 + 0.1 * i) * 1e9, filed=f,
                          adsh=f"e{i:02d}"))
        rows.append(_ni(d, (1 + 0.05 * i) * 1e9, f, adsh=f"n{i:02d}"))
    return rows


def test_pit_later_filing_changes_nothing_before_it():
    base_rows = _quarter_history()
    base = compute(facts=pd.DataFrame(base_rows))
    cutoff = base["avail_date"].max()

    pert_rows = base_rows + [
        # revision of the oldest period, filed years later
        _fact(ddate="2019-03-31", value=99e9, filed="2026-01-05", adsh="zrev",
              period="2025-12-31"),
        _ni("2019-03-31", 88e9, "2026-01-05", adsh="zrevni", period="2025-12-31"),
        # a brand-new quarter
        _fact(ddate="2021-03-31", value=6e9, filed="2021-05-05", adsh="znew"),
        _ni("2021-03-31", 2e9, "2021-05-05", adsh="znewni"),
    ]
    pert = compute(facts=pd.DataFrame(pert_rows))

    old = pert[pert["avail_date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, base)


def test_late_filed_old_period_never_rolls_state_backwards():
    # A delinquent OLD-period equity filed after a newer one must not become
    # the as-of state (the cummax discipline), while a same-day pair keeps
    # the latest period only.
    facts = pd.DataFrame([
        _fact(ddate="2020-06-30", value=7e9, filed="2020-08-01", adsh="b"),
        _fact(ddate="2020-03-31", value=5e9, filed="2020-09-15", adsh="late"),
    ])
    df = compute(facts=facts)
    assert list(df["avail_date"]) == [pd.Timestamp("2020-08-01")]
    assert df.iloc[0]["book_equity"] == 7e9


# --------------------------------------------------------------------------- #
# Panel wiring: B7 mktcap, B8/B9 merges (lag semantics per
# tests/test_availability_grid.py), BRK-B exclusion
# --------------------------------------------------------------------------- #

def _write_market(tmp_path, cal, tickers=("AAA",), close=100.0):
    rows = [{"ticker": t, "date": d, "close": close, "volume": 1e6,
             "ret_21d": 0.01}
            for t in tickers for d in cal]
    pd.DataFrame(rows).to_parquet(tmp_path / "market_data.parquet", index=False)


def _write_shares(tmp_path, rows):
    cols = ["ticker", "cik", "end", "filed", "shares", "concept", "form",
            "class_summed"]
    pd.DataFrame(rows, columns=cols).to_parquet(
        tmp_path / "shares_outstanding_pit.parquet", index=False)


def _share_row(ticker, end, filed, shares,
               concept="dei:EntityCommonStockSharesOutstanding"):
    return {"ticker": ticker, "cik": "0000000001", "end": pd.Timestamp(end),
            "filed": pd.Timestamp(filed), "shares": shares, "concept": concept,
            "form": "10-Q", "class_summed": False}


def test_panel_b7_shares_asof_and_brkb_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=6)
    _write_market(tmp_path, cal, tickers=("AAA", "BRK-B"), close=100.0)
    _write_shares(tmp_path, [
        _share_row("AAA", "2019-12-31", cal[2], 1e9),
        _share_row("BRK-B", "2019-12-31", cal[0], 1.5e6,
                   concept="us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"),
    ])
    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=1)

    aaa = panel[panel["ticker"] == "AAA"].set_index("date")["log_mktcap_pit"]
    # shares_asof semantics: filed <= t, NO extra lag on B7
    assert aaa[cal[:2]].isna().all()
    assert aaa[cal[2:]].eq(np.log(1e9 * 100.0)).all()
    # BRK-B: Class-A-equivalent series -> excluded, NaN throughout
    brk = panel[panel["ticker"] == "BRK-B"]
    assert brk["log_mktcap_pit"].isna().all()
    assert brk["book_to_market_pit"].isna().all()
    assert brk["earnings_yield_pit"].isna().all()


def test_panel_b8_b9_lag_semantics_and_division(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    # Thu, Fri, Mon, Tue around a weekend (the availability-grid pattern)
    cal = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06",
                          "2020-01-07"])
    _write_market(tmp_path, cal, close=100.0)
    _write_shares(tmp_path, [_share_row("AAA", "2019-09-30", "2019-11-01", 1e9)])
    pd.DataFrame([{
        "ticker": "AAA", "avail_date": pd.Timestamp("2020-01-03"),  # Friday
        "book_equity": 20e9, "net_income_ttm": 5e9,
    }]).to_parquet(tmp_path / "fundamentals_pit.parquet", index=False)

    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=1)
    got = panel[panel["ticker"] == "AAA"].set_index("date")

    cap = 1e9 * 100.0
    # Friday avail + t+1 lag -> first usable at the MONDAY close
    for col, num in [("book_to_market_pit", 20e9), ("earnings_yield_pit", 5e9)]:
        assert np.isnan(got.loc[pd.Timestamp("2020-01-02"), col])
        assert np.isnan(got.loc[pd.Timestamp("2020-01-03"), col])
        assert got.loc[pd.Timestamp("2020-01-06"), col] == pytest.approx(num / cap)
        assert got.loc[pd.Timestamp("2020-01-07"), col] == pytest.approx(num / cap)


def test_panel_negative_equity_gives_negative_bm(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=4)
    _write_market(tmp_path, cal, close=50.0)
    _write_shares(tmp_path, [_share_row("AAA", "2019-09-30", "2019-11-01", 2e9)])
    pd.DataFrame([{
        "ticker": "AAA", "avail_date": cal[0],
        "book_equity": -8e9, "net_income_ttm": np.nan,
    }]).to_parquet(tmp_path / "fundamentals_pit.parquet", index=False)

    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=1)
    got = panel[panel["ticker"] == "AAA"].set_index("date")
    assert got.loc[cal[1], "book_to_market_pit"] == pytest.approx(-8e9 / 1e11)
    # NaN TTM never attaches, and must not blank anything else
    assert got["earnings_yield_pit"].isna().all()


def test_panel_missing_caches_all_three_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=4)
    _write_market(tmp_path, cal)
    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=1)
    for col in ["log_mktcap_pit", "book_to_market_pit", "earnings_yield_pit"]:
        assert col in panel.columns
        assert panel[col].isna().all()


# --------------------------------------------------------------------------- #
# Judge: EXTENDED accepted-set shorthand (scripts/ is not a package)
# --------------------------------------------------------------------------- #

def test_judge_extended_is_baseline_plus_b7_b9():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "factor_orthogonality",
        Path(__file__).resolve().parents[1] / "scripts" / "factor_orthogonality.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.EXTENDED == mod.BASELINE + [
        "log_mktcap_pit", "book_to_market_pit", "earnings_yield_pit",
    ]
    # the judge's defaults are UNCHANGED: prior looks stay comparable
    assert "log_mktcap_pit" not in mod.BASELINE
    assert "log_mktcap_pit" not in mod.DEFAULT_CANDIDATES
