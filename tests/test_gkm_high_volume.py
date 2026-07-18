"""C31 gkm_high_volume builder: event classification, warmup gate,
formation-day exclusion from its own reference window."""
import numpy as np
import pandas as pd

from alpha_graph.signals.gkm_high_volume import WINDOW, build_gkm


def panel(ticker, vols, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=len(vols))
    return pd.DataFrame({"ticker": ticker, "date": dates, "volume": vols})


def test_spike_and_drought_classified():
    vols = [100.0] * (WINDOW + 1) + [500.0, 100.0, 1.0, 100.0]
    res = build_gkm(panel("A", vols)).set_index("date")
    dates = pd.bdate_range("2020-01-01", periods=len(vols))
    assert res.loc[dates[WINDOW + 1], "gkm_high_volume"] == 1.0   # spike
    assert res.loc[dates[WINDOW + 3], "gkm_high_volume"] == -1.0  # drought
    assert res.loc[dates[WINDOW + 2], "gkm_high_volume"] == 0.0   # normal


def test_warmup_gate():
    # first emitted day is index WINDOW (needs 49 PRIOR days)
    res = build_gkm(panel("A", [100.0] * (WINDOW + 1)))
    assert len(res) == 1
    assert res["gkm_high_volume"].iloc[0] == 0.0
    assert build_gkm(panel("A", [100.0] * WINDOW)).empty


def test_formation_day_excluded_from_own_reference():
    # A spike day must classify against the PRIOR window only; the next
    # day's reference then contains the spike.
    vols = [100.0] * (WINDOW + 1) + [500.0, 490.0]
    res = build_gkm(panel("A", vols)).set_index("date")
    dates = pd.bdate_range("2020-01-01", periods=len(vols))
    assert res.loc[dates[WINDOW + 1], "gkm_high_volume"] == 1.0
    # 490 vs prior window {100 x 48, 500}: 90th pct interpolates toward 500
    # but 490 > that only if quantile < 490 — with 48 lows and one 500 the
    # 90th percentile is ~100-140 (linear interp), so 490 still classifies +1
    assert res.loc[dates[WINDOW + 2], "gkm_high_volume"] == 1.0
    # flat panels never classify
    flat = build_gkm(panel("A", [100.0] * (WINDOW + 10)))
    assert np.allclose(flat["gkm_high_volume"], 0.0)
