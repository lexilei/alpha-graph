"""Synthetic unit tests for C24 — the 10-Q YoY Lazy Prices min-edit measure and
the freshest-filing combined stream (signals/lazy_prices_edit_10q.py), plus the
judge's additive freshness filter (scripts/factor_orthogonality.py).

Covers: C2's YoY pair enumeration reused verbatim (same-fiscal-quarter ~365d,
NOT adjacent quarters), the pinned digit-free ``[a-z]+`` tokenizer, the
combined-stream same-day dedup keep-last, the ml_combiner loader's
missing-either-cache -> None contract, and the trading-day staleness filter.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from alpha_graph.signals import lazy_prices_edit as lpe
from alpha_graph.signals import lazy_prices_edit_10q as m


# --------------------------------------------------------------------------- #
# 8-quarter fixture: two years of quarterly 10-Qs ~91d apart. Each year-2
# quarter's only ~365d-earlier neighbour is the SAME quarter a year before, so
# the correct YoY pairing is unambiguous and distinct from adjacent quarters.
# --------------------------------------------------------------------------- #

_QUARTER_DATES = [
    "2019-02-15", "2019-05-15", "2019-08-15", "2019-11-15",   # year 1: q0..q3
    "2020-02-15", "2020-05-15", "2020-08-15", "2020-11-15",   # year 2: q4..q7
]
_QUARTER_WORDS = ["alpha", "beta", "gamma", "delta"]


def _eight_quarters(y1_fn, y2_fn) -> list[dict]:
    """Build 8 quarterly 10-Q dicts; y1_fn/y2_fn(word) -> text per quarter."""
    out = []
    for k, d in enumerate(_QUARTER_DATES[:4]):
        out.append({"filing_date": d, "text": y1_fn(_QUARTER_WORDS[k])})
    for k, d in enumerate(_QUARTER_DATES[4:]):
        out.append({"filing_date": d, "text": y2_fn(_QUARTER_WORDS[k])})
    return out


# --------------------------------------------------------------------------- #
# YoY pairing respected (C2's enumeration, pinned by construction)
# --------------------------------------------------------------------------- #

def test_yoy_pairing_matches_same_quarter_one_year_earlier(monkeypatch):
    # distinct text per quarter so nothing about the score can mask a mispair
    filings = _eight_quarters(
        lambda w: f"fiscal 2019 report for the {w} quarter of operations",
        lambda w: f"fiscal 2020 report for the {w} quarter of operations",
    )
    monkeypatch.setattr(m, "_load_filing_texts", lambda t, form: filings)
    rows = m.compute_for_ticker("XYZ")

    # exactly the 4 year-2 quarters get a YoY pair; the 4 year-1 quarters have
    # no ~365d-earlier neighbour and produce nothing.
    assert [r["filing_date"] for r in rows] == _QUARTER_DATES[4:]
    # each pairs with the SAME quarter one year earlier — NOT the adjacent
    # prior quarter (that is the whole point of C2's seasonality control).
    assert [r["prev_filing_date"] for r in rows] == _QUARTER_DATES[:4]
    assert rows[0]["prev_filing_date"] == "2019-02-15"      # not 2019-11-15
    for r in rows:
        assert 300 <= r["gap_days"] <= 430
        assert 0.0 <= r["sim_minedit_10q"] <= 1.0


def test_load_filing_texts_is_called_with_10q_form(monkeypatch):
    seen = {}
    def _spy(ticker, form):
        seen["form"] = form
        return []
    monkeypatch.setattr(m, "_load_filing_texts", _spy)
    m.compute_for_ticker("XYZ")
    assert seen["form"] == "10-Q"


def test_needs_four_filings_for_yoy(monkeypatch):
    # three quarters is below C2's >=4 requirement -> no pairs
    filings = [{"filing_date": d, "text": "some report text here"}
               for d in _QUARTER_DATES[:3]]
    monkeypatch.setattr(m, "_load_filing_texts", lambda t, form: filings)
    assert m.compute_for_ticker("XYZ") == []


# --------------------------------------------------------------------------- #
# Pinned digit-free tokenizer ([a-z]+)
# --------------------------------------------------------------------------- #

def test_tokenizer_mode_is_digit_free_alpha():
    assert m.TOKENIZER_MODE == "alpha"


def test_digit_only_changes_score_one_under_digit_free(monkeypatch):
    # year-2 text differs from its year-1 same-quarter pair ONLY in digits
    # (amount 100->250, year 2019->2020): the digit-free tokenizer sees
    # identical token streams -> sim 1.0 on every YoY pair.
    filings = _eight_quarters(
        lambda w: f"net revenue rose to 100 million in fiscal 2019 {w} segment",
        lambda w: f"net revenue rose to 250 million in fiscal 2020 {w} segment",
    )
    monkeypatch.setattr(m, "_load_filing_texts", lambda t, form: filings)
    rows = m.compute_for_ticker("XYZ")
    assert len(rows) == 4
    assert all(r["sim_minedit_10q"] == pytest.approx(1.0) for r in rows)

    # contrast: keeping digits (C21's alnum tokenizer) the same pair scores < 1
    y1 = "net revenue rose to 100 million in fiscal 2019 alpha segment"
    y2 = "net revenue rose to 250 million in fiscal 2020 alpha segment"
    m_alnum, _, _ = lpe.pair_similarity(lpe.tokenize(y1), lpe.tokenize(y2))
    assert m_alnum < 1.0


# --------------------------------------------------------------------------- #
# Combined freshest-filing stream — same-day dedup keep-last (C7's construction)
# --------------------------------------------------------------------------- #

def test_combine_fresh_stream_same_day_keeps_last_10q():
    # ticker X files a 10-K and a 10-Q on the SAME day 2021-03-01; the 10-Q
    # (concatenated second) must win under keep-last.
    k = pd.DataFrame({"ticker": ["X", "X"],
                      "filing_date": ["2020-03-01", "2021-03-01"],
                      "sim_minedit_alpha_10k": [0.50, 0.60]})
    q = pd.DataFrame({"ticker": ["X", "X"],
                      "filing_date": ["2021-03-01", "2021-06-01"],
                      "sim_minedit_10q": [0.90, 0.80]})
    c = m.combine_fresh_stream(k, q)
    assert list(c.columns) == ["ticker", "filing_date", "sim_minedit_fresh"]
    # union is deduped to 3 distinct (ticker, date) rows, sorted by date
    assert c["filing_date"].tolist() == list(pd.to_datetime(
        ["2020-03-01", "2021-03-01", "2021-06-01"]))
    same_day = c.loc[c["filing_date"] == pd.Timestamp("2021-03-01"),
                     "sim_minedit_fresh"].iloc[0]
    assert same_day == pytest.approx(0.90)      # 10-Q kept, not the 0.60 10-K


def test_combine_fresh_stream_pools_both_forms():
    k = pd.DataFrame({"ticker": ["A"], "filing_date": ["2020-12-20"],
                      "sim_minedit_alpha_10k": [0.7]})
    q = pd.DataFrame({"ticker": ["A", "A"],
                      "filing_date": ["2021-03-05", "2021-06-04"],
                      "sim_minedit_10q": [0.8, 0.85]})
    c = m.combine_fresh_stream(k, q)
    assert len(c) == 3                          # no same-day collision -> all kept
    assert set(c["sim_minedit_fresh"]) == {0.7, 0.8, 0.85}


# --------------------------------------------------------------------------- #
# ml_combiner loader: missing-either-cache -> None (mirrors C7); date carried
# as a value through the as-of merge
# --------------------------------------------------------------------------- #

def _write_min_caches(d: Path):
    pd.DataFrame({"ticker": ["A"], "filing_date": ["2020-12-20"],
                  "sim_minedit_alpha_10k": [0.7]}).to_parquet(
        d / "lazy_prices_edit_alpha.parquet", index=False)
    pd.DataFrame({"ticker": ["A"], "filing_date": ["2021-03-05"],
                  "prev_filing_date": ["2020-03-06"],
                  "sim_minedit_10q": [0.9]}).to_parquet(
        d / "lazy_prices_edit_10q.parquet", index=False)


def test_loader_missing_either_cache_returns_none(tmp_path):
    from alpha_graph.signals.ml_combiner import _load_sim_minedit_fresh
    kpath = tmp_path / "lazy_prices_edit_alpha.parquet"
    # neither cache -> None
    assert _load_sim_minedit_fresh(kpath) is None
    # only the 10-K cache -> None (10-Q sibling missing)
    pd.DataFrame({"ticker": ["A"], "filing_date": ["2020-12-20"],
                  "sim_minedit_alpha_10k": [0.7]}).to_parquet(kpath, index=False)
    assert _load_sim_minedit_fresh(kpath) is None
    # only the 10-Q cache -> None (10-K `path` missing)
    kpath.unlink()
    pd.DataFrame({"ticker": ["A"], "filing_date": ["2021-03-05"],
                  "prev_filing_date": ["2020-03-06"],
                  "sim_minedit_10q": [0.9]}).to_parquet(
        tmp_path / "lazy_prices_edit_10q.parquet", index=False)
    assert _load_sim_minedit_fresh(kpath) is None


def test_loader_both_caches_returns_fresh_with_date_value(tmp_path):
    from alpha_graph.signals.ml_combiner import _load_sim_minedit_fresh
    _write_min_caches(tmp_path)
    out = _load_sim_minedit_fresh(tmp_path / "lazy_prices_edit_alpha.parquet")
    assert list(out.columns) == [
        "ticker", "filing_date", "sim_minedit_fresh", "sim_minedit_fresh_date"]
    # the helper date column equals the filing_date, as a value (not the key)
    assert (out["sim_minedit_fresh_date"] == out["filing_date"]).all()
    assert set(out["sim_minedit_fresh"]) == {0.7, 0.9}


def test_c24_fresh_date_carried_as_value_through_asof(tmp_path, monkeypatch):
    # end-to-end: sim_minedit_fresh attaches at t+1 (filing merge), while
    # sim_minedit_fresh_date carries the TRUE (unlagged) filing_date as a value.
    import alpha_graph.signals.ml_combiner as ml
    monkeypatch.setattr(ml, "CACHE_DIR", tmp_path)
    cal = pd.bdate_range("2020-01-06", periods=8)
    pd.DataFrame([{"ticker": "AAA", "date": d, "close": 100.0 + i,
                   "volume": 1e6, "ret_21d": 0.01}
                  for i, d in enumerate(cal)]).to_parquet(
        tmp_path / "market_data.parquet", index=False)
    pd.DataFrame({"ticker": ["AAA"], "filing_date": [cal[2]],
                  "sim_minedit_alpha_10k": [0.7]}).to_parquet(
        tmp_path / "lazy_prices_edit_alpha.parquet", index=False)
    pd.DataFrame({"ticker": ["AAA"], "filing_date": [cal[5]],
                  "prev_filing_date": [cal[1]],
                  "sim_minedit_10q": [0.9]}).to_parquet(
        tmp_path / "lazy_prices_edit_10q.parquet", index=False)

    panel = ml.build_feature_panel(pit_universe=False, availability_lag_days=1,
                                   daily_signals=True)
    g = panel.set_index("date")
    assert g["sim_minedit_fresh"][cal[:3]].isna().all()       # 10-K unusable t..t
    assert (g["sim_minedit_fresh"][cal[3:5]] == 0.7).all()    # 10-K from t+1
    assert (g["sim_minedit_fresh"][cal[6:]] == 0.9).all()     # 10-Q from t+1
    # the disclosure-date value is the true filing_date (unlagged), carried through
    assert (g["sim_minedit_fresh_date"][cal[3:5]] == cal[2]).all()
    assert (g["sim_minedit_fresh_date"][cal[6:]] == cal[5]).all()


# --------------------------------------------------------------------------- #
# Judge freshness filter (scripts/factor_orthogonality.py) — additive
# --------------------------------------------------------------------------- #

def _load_judge():
    spec = importlib.util.spec_from_file_location(
        "fo_c24",
        Path(__file__).resolve().parents[1] / "scripts" / "factor_orthogonality.py")
    fo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fo)
    return fo


def test_trading_day_staleness_counts_sessions():
    fo = _load_judge()
    cal = pd.bdate_range("2020-01-06", periods=10).values  # d0..d9
    rows = pd.Series([cal[9], cal[9], cal[9]])
    disc = pd.Series([cal[4], cal[9], pd.NaT])
    stale = fo._trading_day_staleness(rows, disc, cal)
    assert stale[0] == 5.0        # sessions in (d4, d9]: d5,d6,d7,d8,d9
    assert stale[1] == 0.0        # same session
    assert stale[2] == -1.0       # NaT sentinel


def test_apply_freshness_filter_fresh_and_stale_complement():
    fo = _load_judge()
    cal = pd.bdate_range("2020-01-06", periods=10).values
    pm = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "month": pd.PeriodIndex(["2020-01"] * 4, freq="M"),
        "date": [cal[9]] * 4,
        "cand": [1.0, 2.0, 3.0, 4.0],
        "disc": [cal[4], cal[0], pd.NaT, cal[8]],   # stale 5, 9, NaT, 1
    })
    fresh = fo.apply_freshness_filter(pm, "disc", cal, max_days=5)
    assert set(fresh["ticker"]) == {"A", "D"}       # <=5 td old; C (NaT) dropped
    stale = fo.apply_freshness_filter(pm, "disc", cal, min_days=6)
    assert set(stale["ticker"]) == {"B"}            # >=6 td old
    both = fo.apply_freshness_filter(pm, "disc", cal, max_days=5, min_days=1)
    assert set(both["ticker"]) == {"A", "D"}


def test_apply_freshness_filter_missing_col_raises():
    fo = _load_judge()
    cal = pd.bdate_range("2020-01-06", periods=5).values
    pm = pd.DataFrame({"date": [cal[4]], "month": pd.PeriodIndex(["2020-01"],
                                                                 freq="M")})
    with pytest.raises(ValueError):
        fo.apply_freshness_filter(pm, "nope", cal, max_days=5)
