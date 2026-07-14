"""Tests for scripts/repair_market_data.py (2026-07-14 price-panel repair).

Two layers: synthetic tests of the pure repair steps (always run, loaded by
path — scripts/ is not a package), and post-repair invariants on the real
cache (skipped when the repaired cache is absent).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_graph.config import CACHE_DIR

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_market_data.py"
_spec = importlib.util.spec_from_file_location("repair_market_data", _SCRIPT)
repair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair)

MARKET_PATH = CACHE_DIR / "market_data.parquet"
SEAMS_PATH = CACHE_DIR / "market_data_seams.parquet"

needs_repaired = pytest.mark.skipif(
    not (MARKET_PATH.exists() and SEAMS_PATH.exists()),
    reason="repaired market cache not present",
)


# --------------------------------------------------------------------------- #
# Synthetic: seam masking
# --------------------------------------------------------------------------- #

def _toy_panel(n=60, tickers=("AAA", "BBB")) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    frames = []
    for k, tk in enumerate(tickers):
        close = pd.Series(np.arange(1, n + 1, dtype=float) + 100 * k)
        f = pd.DataFrame({"ticker": tk, "date": dates, "close": close})
        for h, col in repair.RET_HORIZONS:
            f[col] = close.pct_change(h).shift(-h)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_mask_seam_returns_exact_window():
    df = _toy_panel()
    seam_pos = 30
    seam_date = df.loc[df["ticker"] == "AAA", "date"].iloc[seam_pos]
    out, n_cells = repair.mask_seam_returns(
        df, seams=[("AAA", str(seam_date.date()), "test")])
    assert n_cells == 1 + 5 + 21

    a = out[out["ticker"] == "AAA"].reset_index(drop=True)
    for h, col in repair.RET_HORIZONS:
        vals = a[col]
        nan_pos = set(np.flatnonzero(vals.isna().to_numpy()))
        expected = set(range(seam_pos - h, seam_pos))
        # forward returns at the tail are NaN by construction
        tail = set(range(len(a) - h, len(a)))
        assert nan_pos == expected | tail, col
    # the seam-day row itself keeps its forward returns
    assert not a.loc[seam_pos, ["ret_1d", "ret_5d", "ret_21d"]].isna().any()
    # the other ticker is untouched
    b = out[out["ticker"] == "BBB"].reset_index(drop=True)
    assert int(b["ret_21d"].isna().sum()) == 21  # tail-only


def test_mask_seam_returns_idempotent_and_missing_date():
    df = _toy_panel()
    seam_date = str(df.loc[df["ticker"] == "AAA", "date"].iloc[30].date())
    once, n1 = repair.mask_seam_returns(df, seams=[("AAA", seam_date, "t")])
    twice, n2 = repair.mask_seam_returns(once, seams=[("AAA", seam_date, "t")])
    assert n1 == 27 and n2 == 0
    pd.testing.assert_frame_equal(once, twice)
    # a seam date not in the panel is skipped, not an error
    _, n3 = repair.mask_seam_returns(df, seams=[("AAA", "1999-01-01", "t")])
    assert n3 == 0


# --------------------------------------------------------------------------- #
# Synthetic: identity/synthetic drops
# --------------------------------------------------------------------------- #

def test_drop_wrong_identity_and_synthetic():
    df = pd.DataFrame({
        "ticker": ["SW", "SW", "AMCR", "VRT", "GDDY", "GDDY", "AAPL"],
        "date": pd.to_datetime([
            "2024-07-05", "2024-07-08",       # SW: first is Smurfit Kappa OTC
            "2019-06-10",                     # AMCR pre-listing
            "2020-02-07",                     # VRT SPAC shell
            "2015-03-31", "2015-04-01",       # GDDY synthetic + real
            "2015-03-31",
        ]),
        "close": 1.0,
    })
    out, dropped = repair.drop_wrong_identity(df)
    assert dropped == {"SW": 1, "AMCR": 1, "VRT": 1}
    out, n_synth = repair.drop_synthetic_rows(out)
    assert n_synth == 1
    kept = set(zip(out["ticker"], out["date"].dt.strftime("%Y-%m-%d")))
    assert kept == {("SW", "2024-07-08"), ("GDDY", "2015-04-01"),
                    ("AAPL", "2015-03-31")}
    # idempotent
    again, dropped2 = repair.drop_wrong_identity(out)
    assert dropped2 == {"SW": 0, "AMCR": 0, "VRT": 0}
    assert len(again) == len(out)


# --------------------------------------------------------------------------- #
# Real repaired cache: invariants
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def market():
    return pd.read_parquet(
        MARKET_PATH,
        columns=["ticker", "date", "close", "ret_1d", "ret_5d", "ret_21d"])


@needs_repaired
def test_seams_parquet_schema():
    seams = pd.read_parquet(SEAMS_PATH)
    assert list(seams.columns) == ["ticker", "seam_date", "reason"]
    assert len(seams) == 6
    pairs = set(zip(seams["ticker"],
                    pd.to_datetime(seams["seam_date"]).dt.strftime("%Y-%m-%d")))
    assert pairs == {("DHR", "2016-07-05"), ("EXPE", "2011-12-21"),
                     ("DD", "2019-06-03"), ("LDOS", "2013-09-30"),
                     ("HWM", "2020-04-01"), ("HPQ", "2015-11-02")}


@needs_repaired
def test_wrong_identity_prefixes_absent(market):
    for ticker, first_valid in repair.WRONG_IDENTITY_PREFIXES.items():
        sub = market.loc[market["ticker"] == ticker, "date"]
        assert not sub.empty, ticker
        assert sub.min() >= pd.Timestamp(first_valid), ticker


@needs_repaired
def test_gddy_synthetic_row_absent(market):
    g = market[market["ticker"] == "GDDY"]
    assert pd.Timestamp("2015-03-31") not in set(g["date"])
    assert pd.Timestamp("2015-04-01") in set(g["date"])


@needs_repaired
def test_seam_crossing_returns_nan(market):
    seams = pd.read_parquet(SEAMS_PATH)
    for row in seams.itertuples():
        sub = (market[market["ticker"] == row.ticker]
               .sort_values("date").reset_index(drop=True))
        pos = int(sub.index[sub["date"] == pd.Timestamp(row.seam_date)][0])
        # every window crossing the seam is NaN...
        for h, col in repair.RET_HORIZONS:
            assert sub.loc[pos - h:pos - 1, col].isna().all(), (row.ticker, col)
        # ...the first non-crossing window on each side is not
        assert not np.isnan(sub.loc[pos - 22, "ret_21d"]), row.ticker
        assert not np.isnan(sub.loc[pos, "ret_1d"]), row.ticker
        # prices themselves are untouched (the seam day still exists)
        assert not np.isnan(sub.loc[pos, "close"]), row.ticker


@needs_repaired
def test_missing_tickers_readded(market):
    for ticker in repair.MISSING_TICKERS:
        sub = market[market["ticker"] == ticker]
        assert len(sub) > 250, ticker
        assert sub["ret_21d"].notna().sum() > 200, ticker
