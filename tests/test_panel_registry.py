"""Tests for the FACTOR_SOURCES panel registry: schema sanity and
end-to-end availability-lag semantics per merge kind through
build_feature_panel (plan task #5, registry-driven panel builder).

Conventions under test (see the FactorSource docstring in ml_combiner):
  - "filing" sources lag by availability_lag_days on EVERY grid;
  - "grid" sources lag ONLY under daily_signals (a +1d shift on a
    month-end-stamped series is the stale variant, not t+1);
  - "grid_static" sources never lag (C8/C9, monthly C11);
  - daily_filename switches the cache under daily_signals (C10);
  - daily_only sources are absent from the monthly-grid panel (C15);
  - a missing cache yields an all-NaN column plus a debug log, never a crash.
"""

import numpy as np
import pandas as pd
import pytest

import alpha_graph.signals.ml_combiner as ml
from alpha_graph.signals.ml_combiner import (
    FACTOR_SOURCES,
    VALID_KINDS,
    FactorSource,
)


# --------------------------------------------------------------------------- #
# Registry schema sanity
# --------------------------------------------------------------------------- #

def test_registry_kinds_valid():
    assert all(s.kind in VALID_KINDS for s in FACTOR_SOURCES)


def test_registry_invalid_kind_or_dropna_rejected_at_construction():
    with pytest.raises(ValueError):
        FactorSource("X", "x.parquet", ("x",), "date", "monthly")
    with pytest.raises(ValueError):
        FactorSource("X", "x.parquet", ("x",), "date", "grid", dropna="some")


def test_registry_no_duplicate_columns():
    cols = [c for s in FACTOR_SOURCES for c in s.cols]
    assert len(cols) == len(set(cols))


def test_registry_columns_do_not_collide_with_panel_base():
    base = {
        "ticker", "date", "close", "volume", "fwd_return_21d", "sector",
        "momentum_21d", "momentum_5d", "momentum_252_21", "volatility_21d",
        "volume_zscore", "log_dollar_volume", "log_mktcap_pit",
        "book_to_market_pit", "earnings_yield_pit", "book_equity",
        "net_income_ttm", "_mktcap_pit", "_shares_pit", "daily_ret",
    }
    for s in FACTOR_SOURCES:
        assert not (set(s.cols) & base), s.label


def test_registry_v0_columns_all_registered():
    # The 13 v0 signal columns must stay registered (supersets are fine —
    # new factors append rows; they never displace these).
    cols = {c for s in FACTOR_SOURCES for c in s.cols}
    assert {
        "cosine_similarity", "cos_10q_yoy", "embed_sim_10k_fin",
        "tone_shift_10k", "embed_sim_10k_bge", "new_content_frac",
        "cos_latest_filing", "spillover_event", "spillover_momentum",
        "spillover_cust_mom", "evt8k_freq_z", "evt8k_freq_z_d", "sue_pead",
    } <= cols


def test_registry_field_shapes():
    for s in FACTOR_SOURCES:
        assert s.filename.endswith(".parquet"), s.label
        assert isinstance(s.cols, tuple) and len(s.cols) >= 1, s.label
        assert all(isinstance(c, str) for c in s.cols), s.label
        assert s.date_col, s.label
        if s.daily_filename is not None:
            assert s.daily_filename != s.filename, s.label
        if s.daily_only:
            # daily-only sources have a single (daily) cache
            assert s.daily_filename is None, s.label


# --------------------------------------------------------------------------- #
# End-to-end through build_feature_panel (tiny synthetic caches)
# --------------------------------------------------------------------------- #

def _write_market(tmp_path, cal, tickers=("AAA",)):
    rows = [{"ticker": t, "date": d, "close": 100.0 + i, "volume": 1e6,
             "ret_21d": 0.01}
            for t in tickers for i, d in enumerate(cal)]
    pd.DataFrame(rows).to_parquet(tmp_path / "market_data.parquet", index=False)


def test_missing_caches_yield_all_nan_columns(tmp_path, monkeypatch):
    # CACHE_DIR holding ONLY market data: every registered column must exist
    # and be all-NaN — the exist-or-NaN contract for all sources at once,
    # including the loader-based ones (C7, C17).
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=5)
    _write_market(tmp_path, cal)
    panel = ml.build_feature_panel(pit_universe=False)
    for s in FACTOR_SOURCES:
        for c in s.cols:
            assert c in panel.columns, (s.label, c)
            assert panel[c].isna().all(), (s.label, c)


def test_filing_kind_lags_on_monthly_grid_too(tmp_path, monkeypatch):
    # "filing" semantics: availability_lag_days applies on EVERY grid —
    # daily_signals=False must still push a filing to the next close.
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=8)   # Mon..Wed, no weekend gap
    _write_market(tmp_path, cal)
    pd.DataFrame({"ticker": ["AAA"], "filing_date": [cal[2]],
                  "cosine_similarity": [0.7]}).to_parquet(
        tmp_path / "lazy_prices_signal_10K.parquet", index=False)
    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=1,
                                   daily_signals=False)
    got = panel.set_index("date")["cosine_similarity"]
    assert got[cal[:3]].isna().all()        # unusable at the filing close
    assert (got[cal[3:]] == 0.7).all()      # first use next close, carried fwd


def test_filing_kind_lag_zero_attaches_same_day(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=5)
    _write_market(tmp_path, cal)
    pd.DataFrame({"ticker": ["AAA"], "filing_date": [cal[2]],
                  "cosine_similarity": [0.7]}).to_parquet(
        tmp_path / "lazy_prices_signal_10K.parquet", index=False)
    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=0,
                                   daily_signals=False)
    got = panel.set_index("date")["cosine_similarity"]
    assert got[cal[:2]].isna().all()
    assert (got[cal[2:]] == 0.7).all()      # same-close attach (lag 0)


def test_grid_kind_daily_switch_and_lag(tmp_path, monkeypatch):
    # C10 end-to-end: daily_signals picks the DAILY cache and applies the
    # lag; the monthly grid loads the monthly cache UNLAGGED (a month-end
    # series cannot express t+1). Different values in each cache make the
    # file switch observable.
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=8)
    _write_market(tmp_path, cal)
    pd.DataFrame({"ticker": ["AAA"], "date": [cal[2]],
                  "spillover_cust_mom": [0.5]}).to_parquet(
        tmp_path / "graph_customer_momentum.parquet", index=False)
    pd.DataFrame({"ticker": ["AAA"], "date": [cal[2]],
                  "spillover_cust_mom": [0.9]}).to_parquet(
        tmp_path / "graph_customer_momentum_daily.parquet", index=False)

    daily = ml.build_feature_panel(pit_universe=False, availability_lag_days=1,
                                   daily_signals=True)
    got = daily.set_index("date")["spillover_cust_mom"]
    assert got[cal[:3]].isna().all()        # t+1: the stamp close is unusable
    assert (got[cal[3:]] == 0.9).all()      # daily-cache value, next close

    monthly = ml.build_feature_panel(pit_universe=False,
                                     availability_lag_days=1,
                                     daily_signals=False)
    got_m = monthly.set_index("date")["spillover_cust_mom"]
    assert got_m[cal[:2]].isna().all()
    assert (got_m[cal[2:]] == 0.5).all()    # monthly cache, unlagged


def test_grid_kind_dropna_nan_rows_never_blank_carry_forward(tmp_path,
                                                             monkeypatch):
    # C10's caches carry NaN rows (no qualifying customers that day); they
    # are dropped BEFORE the as-of merge so they never overwrite a real value.
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=6)
    _write_market(tmp_path, cal)
    pd.DataFrame({"ticker": ["AAA", "AAA"], "date": [cal[1], cal[3]],
                  "spillover_cust_mom": [0.5, np.nan]}).to_parquet(
        tmp_path / "graph_customer_momentum_daily.parquet", index=False)
    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=0,
                                   daily_signals=True)
    got = panel.set_index("date")["spillover_cust_mom"]
    assert (got[cal[1:]] == 0.5).all()      # NaN row dropped, value carried


def test_grid_static_never_lagged(tmp_path, monkeypatch):
    # C11: always merged unlagged — the month-m z attaches at month-end m
    # even when availability_lag_days=1 and daily_signals=True.
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=6)
    _write_market(tmp_path, cal)
    pd.DataFrame({"ticker": ["AAA"], "date": [cal[2]],
                  "evt8k_freq_z": [1.5]}).to_parquet(
        tmp_path / "event_freq_8k.parquet", index=False)
    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=1,
                                   daily_signals=True)
    got = panel.set_index("date")["evt8k_freq_z"]
    assert got[cal[:2]].isna().all()
    assert (got[cal[2:]] == 1.5).all()      # attaches at its stamp date


def test_daily_only_column_absent_on_monthly_grid(tmp_path, monkeypatch):
    # C15 has no monthly counterpart: under daily_signals=False the column
    # is ABSENT (not NaN) — the historical monthly panel shape is preserved.
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=5)
    _write_market(tmp_path, cal)
    pd.DataFrame({"ticker": ["AAA"], "date": [cal[1]],
                  "evt8k_freq_z_d": [2.0]}).to_parquet(
        tmp_path / "event_freq_8k_daily.parquet", index=False)
    daily = ml.build_feature_panel(pit_universe=False, daily_signals=True)
    assert "evt8k_freq_z_d" in daily.columns
    assert (daily.set_index("date")["evt8k_freq_z_d"][cal[2:]] == 2.0).all()
    monthly = ml.build_feature_panel(pit_universe=False, daily_signals=False)
    assert "evt8k_freq_z_d" not in monthly.columns
