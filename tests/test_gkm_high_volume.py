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
    # 490 vs prior window {100 x 48, 500}: the 90th percentile of 49 obs
    # interpolates between order stats 44 and 45 — both 100 here — so
    # q_hi is exactly 100 and 490 still classifies +1 (audit-corrected)
    assert res.loc[dates[WINDOW + 2], "gkm_high_volume"] == 1.0
    # flat panels never classify
    flat = build_gkm(panel("A", [100.0] * (WINDOW + 10)))
    assert np.allclose(flat["gkm_high_volume"], 0.0)


def test_multi_ticker_no_boundary_leakage():
    """Cross-ticker isolation: a huge-volume ticker adjacent in frame order
    must not leak into its neighbor's reference window via shift/rolling."""
    a = panel("A", [100.0] * (WINDOW + 2))
    z = panel("Z", [1e9] * (WINDOW + 2))          # sorts after A
    both = build_gkm(pd.concat([a, z], ignore_index=True))
    solo = build_gkm(a)
    a_both = both[both["ticker"] == "A"].reset_index(drop=True)
    pd.testing.assert_frame_equal(a_both, solo)
    # Z's flat (huge) volumes classify 0, not -1 against A's small window
    assert np.allclose(both[both["ticker"] == "Z"]["gkm_high_volume"], 0.0)
