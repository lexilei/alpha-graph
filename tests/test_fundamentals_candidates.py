"""Tests for C25/C26/C27 (synthetic inputs only — no real caches touched).

Pinned semantics under test (registration 2026-07-15, commit `28b84c6`):
  - C25 net_issuance_12m: first-filed share count per (ticker, end), 12m log
    change matched ~365+/-45d earlier, availability = the LATER obs's filed
    date; the split rule NaNs any 12m window containing a >1.9x/<0.55x
    consecutive count step; BRK-B excluded;
  - C26 asset_growth_yoy: YoY log change in first-filed as-filed Assets
    (qtrs=0), matched ~365+/-45d earlier, availability = the later accession's
    filed date; first-filed beats a later restatement;
  - C27 ann_ret_2d: announcement day a = acceptance date if before 16:00 ET
    else the next trading day; signal = close(a-1)->close(a+1); availability =
    a+1; same-day multi-8-K keeps the FIRST 2.02 of the day per ticker;
  - all three merge through build_feature_panel with v0 t+1 lag semantics.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.signals.fundamentals_candidates import (
    compute_ann_ret_2d,
    compute_asset_growth,
    compute_net_issuance,
)


# --------------------------------------------------------------------------- #
# C25 net_issuance_12m
# --------------------------------------------------------------------------- #

def _shares(ticker, ends, shares, filed_lag_days=35):
    ends = pd.to_datetime(list(ends))
    return pd.DataFrame({
        "ticker": ticker, "end": ends,
        "filed": ends + pd.Timedelta(days=filed_lag_days),
        "shares": list(map(float, shares)),
    })


def test_c25_12m_match_and_availability():
    ends = pd.date_range("2019-03-31", periods=8, freq="QE")   # 2019Q1..2020Q4
    shr = [100, 101, 102, 103, 104, 105, 106, 107]
    out = compute_net_issuance(_shares("AAA", ends, shr)).set_index("end")
    # 2020-06-30 (idx 5) matches 2019-06-30 (idx 1): log(105/101)
    got = out.loc[ends[5], "net_issuance_12m"]
    assert got == pytest.approx(np.log(105 / 101), abs=1e-12)
    # availability = the LATER (current) observation's filed date
    assert out.loc[ends[5], "avail_date"] == ends[5] + pd.Timedelta(days=35)
    # first year has no ~365d-earlier match -> NaN
    assert np.isnan(out.loc[ends[0], "net_issuance_12m"])


def test_c25_seasonal_match_skips_gap_year():
    # a 2019-06-30 obs and a 2021-06-30 obs (730d): must NOT match across 2y
    out = compute_net_issuance(
        _shares("AAA", ["2019-06-30", "2021-06-30"], [100, 150])
    ).set_index("end")
    assert np.isnan(out.loc[pd.Timestamp("2021-06-30"), "net_issuance_12m"])


def test_c25_split_step_nans_the_window():
    # a ~2x split step at 2020-03-31 (103 -> 206) must NaN every 12m window
    # that contains it, but a window fully AFTER the split is defined.
    ends = pd.date_range("2019-03-31", periods=12, freq="QE")   # 2019Q1..2021Q4
    shr = [100, 101, 102, 103, 206, 208, 210, 212, 214, 216, 218, 220]
    out = compute_net_issuance(_shares("SPL", ends, shr)).set_index("end")
    # 2020-06-30 (idx 5) window (idx1, idx5] contains the split step at idx4
    assert np.isnan(out.loc[ends[5], "net_issuance_12m"])
    assert bool(out.loc[ends[5], "split_nan"])
    # 2021-03-31 (idx 8) window (idx4, idx8] is fully post-split -> defined
    assert out.loc[ends[8], "net_issuance_12m"] == pytest.approx(
        np.log(214 / 206), abs=1e-12)
    assert not bool(out.loc[ends[8], "split_nan"])


def test_c25_reverse_split_step_also_flagged():
    # a <0.55x step (1:2 reverse split, 200 -> 90) is flagged the same way
    ends = pd.date_range("2019-03-31", periods=8, freq="QE")
    shr = [200, 200, 200, 200, 90, 90, 90, 90]
    out = compute_net_issuance(_shares("REV", ends, shr)).set_index("end")
    assert bool(out.loc[ends[4], "split_nan"])
    assert np.isnan(out.loc[ends[4], "net_issuance_12m"])


def test_c25_excludes_brk_b():
    ends = pd.date_range("2019-03-31", periods=8, freq="QE")
    frame = pd.concat([
        _shares("AAA", ends, [100, 101, 102, 103, 104, 105, 106, 107]),
        _shares("BRK-B", ends, [1, 1, 1, 1, 1, 1, 1, 1]),
    ], ignore_index=True)
    out = compute_net_issuance(frame)
    assert "BRK-B" not in set(out["ticker"])
    assert "AAA" in set(out["ticker"])


def test_c25_first_filed_per_end_wins():
    # same (ticker, end) reported by two filings: the earlier-filed count wins
    ends = pd.date_range("2019-03-31", periods=8, freq="QE")
    base = _shares("AAA", ends, [100, 101, 102, 103, 104, 105, 106, 107])
    # a late correction of the 2020-06-30 count to 999, filed a year later
    late = pd.DataFrame({"ticker": ["AAA"], "end": [ends[5]],
                         "filed": [ends[5] + pd.Timedelta(days=400)],
                         "shares": [999.0]})
    out = compute_net_issuance(pd.concat([base, late], ignore_index=True)
                               ).set_index("end")
    # uses the first-filed 105, not the 999 correction
    assert out.loc[ends[5], "net_issuance_12m"] == pytest.approx(
        np.log(105 / 101), abs=1e-12)


# --------------------------------------------------------------------------- #
# C26 asset_growth_yoy
# --------------------------------------------------------------------------- #

def _afact(cik, ticker, ddate, value, filed, adsh, qtrs=0, tag="Assets",
           uom="USD", accepted=None):
    filed = pd.Timestamp(filed)
    return dict(cik=cik, ticker=ticker, tag=tag, uom=uom, qtrs=qtrs,
                ddate=pd.Timestamp(ddate), filed=filed,
                accepted=pd.Timestamp(accepted) if accepted else filed,
                adsh=adsh, value=float(value), form="10-Q")


def test_c26_yoy_match_and_availability():
    facts = pd.DataFrame([
        _afact("1", "AAA", "2019-06-30", 100.0, "2019-08-01", "a1"),
        _afact("1", "AAA", "2020-06-30", 130.0, "2020-08-01", "a2"),
    ])
    out = compute_asset_growth(facts).set_index("ddate")
    assert out.loc[pd.Timestamp("2020-06-30"), "asset_growth_yoy"] == pytest.approx(
        np.log(130 / 100), abs=1e-12)
    # availability = the later accession's filed date
    assert out.loc[pd.Timestamp("2020-06-30"), "avail_date"] == pd.Timestamp("2020-08-01")
    assert np.isnan(out.loc[pd.Timestamp("2019-06-30"), "asset_growth_yoy"])


def test_c26_first_filed_beats_restatement():
    facts = pd.DataFrame([
        _afact("1", "AAA", "2019-06-30", 100.0, "2019-08-01", "orig"),
        # next year's comparative restates 2019-06-30 to 999 — must lose
        _afact("1", "AAA", "2019-06-30", 999.0, "2020-08-01", "restate"),
        _afact("1", "AAA", "2020-06-30", 130.0, "2020-08-01", "a2"),
    ])
    out = compute_asset_growth(facts).set_index("ddate")
    assert out.loc[pd.Timestamp("2020-06-30"), "asset_growth_yoy"] == pytest.approx(
        np.log(130 / 100), abs=1e-12)


def test_c26_ignores_non_instant_and_other_tags():
    facts = pd.DataFrame([
        _afact("1", "AAA", "2019-06-30", 100.0, "2019-08-01", "a1"),
        _afact("1", "AAA", "2020-06-30", 130.0, "2020-08-01", "a2"),
        # a qtrs=4 Assets row and a Liabilities row must be ignored entirely
        _afact("1", "AAA", "2020-06-30", 5.0, "2020-08-01", "junk1", qtrs=4),
        _afact("1", "AAA", "2020-06-30", 7.0, "2020-08-01", "junk2", tag="Liabilities"),
    ])
    out = compute_asset_growth(facts)
    # exactly one row per (ticker, ddate) instant, YoY uses the qtrs=0 values
    assert len(out) == 2
    assert out.set_index("ddate").loc[pd.Timestamp("2020-06-30"),
                                      "asset_growth_yoy"] == pytest.approx(
        np.log(130 / 100), abs=1e-12)


# --------------------------------------------------------------------------- #
# C27 ann_ret_2d
# --------------------------------------------------------------------------- #

def _market(cal, ticker="AAA"):
    return pd.DataFrame({"ticker": ticker, "date": cal,
                         "close": [100.0 + i for i in range(len(cal))]})


def _item(accession, ticker, filing_date, items="2.02,9.01"):
    return dict(accession=accession, ticker=ticker,
                filing_date=pd.Timestamp(filing_date), items=items, n_items=2)


def _acc(accession, ts):
    return dict(accession=accession, acceptance_ts=pd.Timestamp(ts))


def test_c27_before_1600_same_day_and_arithmetic():
    cal = pd.bdate_range("2020-01-06", periods=6)          # Mon..Mon (no holidays)
    items = pd.DataFrame([_item("e1", "AAA", cal[2])])
    acc = pd.DataFrame([_acc("e1", cal[2] + pd.Timedelta(hours=15, minutes=59))])
    out = compute_ann_ret_2d(items, acc, _market(cal))
    assert len(out) == 1
    r = out.iloc[0]
    assert r["ann_day"] == cal[2]                          # accepted pre-16:00
    assert r["avail_date"] == cal[3]                       # availability = a+1
    # close(a-1)=close(cal[1])=101, close(a+1)=close(cal[3])=103
    assert r["ann_ret_2d"] == pytest.approx(103 / 101 - 1, abs=1e-12)


def test_c27_at_1600_shifts_to_next_trading_day():
    cal = pd.bdate_range("2020-01-06", periods=6)
    items = pd.DataFrame([_item("e1", "AAA", cal[2])])
    acc = pd.DataFrame([_acc("e1", cal[2] + pd.Timedelta(hours=16))])
    out = compute_ann_ret_2d(items, acc, _market(cal))
    r = out.iloc[0]
    assert r["ann_day"] == cal[3]                          # 16:00 -> next trading day
    assert r["avail_date"] == cal[4]                       # a+1
    # close(a-1)=close(cal[2])=102, close(a+1)=close(cal[4])=104
    assert r["ann_ret_2d"] == pytest.approx(104 / 102 - 1, abs=1e-12)


def test_c27_after_close_friday_rolls_over_weekend():
    # Friday 16:30 acceptance -> announcement day is the following Monday
    cal = pd.bdate_range("2020-01-06", periods=8)          # spans a weekend
    fri = cal[4]                                            # Fri 2020-01-10
    assert fri.dayofweek == 4
    items = pd.DataFrame([_item("e1", "AAA", fri)])
    acc = pd.DataFrame([_acc("e1", fri + pd.Timedelta(hours=16, minutes=30))])
    out = compute_ann_ret_2d(items, acc, _market(cal))
    assert out.iloc[0]["ann_day"] == cal[5]                # Monday
    assert out.iloc[0]["avail_date"] == cal[6]


def test_c27_same_day_dedup_keeps_first_2_02():
    # two 2.02 8-Ks the same filing day: the EARLIER-accepted one wins, so the
    # announcement day is its pre-16:00 day (not the later 16:30 filing's a+1)
    cal = pd.bdate_range("2020-01-06", periods=6)
    items = pd.DataFrame([_item("e_early", "AAA", cal[2]),
                          _item("e_late", "AAA", cal[2])])
    acc = pd.DataFrame([_acc("e_early", cal[2] + pd.Timedelta(hours=9)),
                        _acc("e_late", cal[2] + pd.Timedelta(hours=16, minutes=30))])
    out = compute_ann_ret_2d(items, acc, _market(cal))
    assert len(out) == 1
    assert out.iloc[0]["ann_day"] == cal[2]                # the 09:00 event, pre-16:00


def test_c27_non_2_02_ignored_and_missing_close_skipped():
    cal = pd.bdate_range("2020-01-06", periods=6)
    items = pd.DataFrame([
        _item("e1", "AAA", cal[2], items="5.02,9.01"),     # not a 2.02 -> ignored
        _item("e2", "BBB", cal[2]),                        # 2.02 but no prices
    ])
    acc = pd.DataFrame([_acc("e1", cal[2] + pd.Timedelta(hours=8)),
                        _acc("e2", cal[2] + pd.Timedelta(hours=8))])
    out = compute_ann_ret_2d(items, acc, _market(cal, ticker="AAA"))
    assert out.empty


# --------------------------------------------------------------------------- #
# Panel merge (build_feature_panel C25/C26/C27 block, v0 t+1 lag)
# --------------------------------------------------------------------------- #

def _write_market(tmp_path, cal, tickers=("AAA",)):
    rows = [{"ticker": t, "date": d, "close": 100.0 + i, "volume": 1e6,
             "ret_21d": 0.01}
            for t in tickers for i, d in enumerate(cal)]
    pd.DataFrame(rows).to_parquet(tmp_path / "market_data.parquet", index=False)


def test_panel_merge_all_three_v0_lag(tmp_path, monkeypatch):
    import alpha_graph.signals.ml_combiner as ml
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=8)
    _write_market(tmp_path, cal)
    pd.DataFrame([{"ticker": "AAA", "avail_date": cal[2], "end": cal[2],
                   "net_issuance_12m": 0.5, "split_nan": False},
                  # a later split-NaN'd row must NOT blank the carried value
                  {"ticker": "AAA", "avail_date": cal[5], "end": cal[5],
                   "net_issuance_12m": np.nan, "split_nan": True}]
                 ).to_parquet(tmp_path / "net_issuance_12m.parquet", index=False)
    pd.DataFrame([{"ticker": "AAA", "avail_date": cal[2], "ddate": cal[2],
                   "asset_growth_yoy": 0.3}]
                 ).to_parquet(tmp_path / "asset_growth_yoy.parquet", index=False)
    pd.DataFrame([{"ticker": "AAA", "avail_date": cal[2], "ann_day": cal[1],
                   "ann_ret_2d": 0.2}]
                 ).to_parquet(tmp_path / "ann_ret_2d.parquet", index=False)

    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=1)
    for col, val in [("net_issuance_12m", 0.5), ("asset_growth_yoy", 0.3),
                     ("ann_ret_2d", 0.2)]:
        got = panel[panel["ticker"] == "AAA"].set_index("date")[col]
        assert got[cal[:3]].isna().all()        # v0 t+1: unusable through the avail close
        assert (got[cal[3:]] == val).all()       # attaches next close, carried forward


def test_panel_merge_missing_caches_are_nan(tmp_path, monkeypatch):
    import alpha_graph.signals.ml_combiner as ml
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=5)
    _write_market(tmp_path, cal)
    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=1)
    for col in ["net_issuance_12m", "asset_growth_yoy", "ann_ret_2d"]:
        assert col in panel.columns
        assert panel[col].isna().all()
