"""Tests for ff_attribution — synthetic factors only, no network, no parquet."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from alpha_graph.eval.ff_attribution import (
    DEFAULT_HAC_LAGS,
    MODELS,
    ff_attribution,
)

BETA_MKT, BETA_HML, ALPHA = 0.5, -0.2, 3e-4  # planted: 3 bps/day alpha


# --------------------------------------------------------------------------- #
# Synthetic factor panel + pnl with planted loadings
# --------------------------------------------------------------------------- #

def synth_factors(n=2000, seed=11):
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    return pd.DataFrame({
        "mkt_rf": rng.normal(4e-4, 0.010, n),
        "smb": rng.normal(0.0, 0.005, n),
        "hml": rng.normal(0.0, 0.006, n),
        "rmw": rng.normal(0.0, 0.004, n),
        "cma": rng.normal(0.0, 0.004, n),
        "mom": rng.normal(0.0, 0.007, n),
        "rf": np.full(n, 8e-5),
    }, index=idx)


def synth_pnl(f, noise=0.002, seed=5):
    rng = np.random.RandomState(seed)
    return (f["rf"] + BETA_MKT * f["mkt_rf"] + BETA_HML * f["hml"] + ALPHA
            + noise * rng.normal(size=len(f)))


# --------------------------------------------------------------------------- #
# Recovery of planted coefficients
# --------------------------------------------------------------------------- #

def test_recovers_planted_betas_and_alpha():
    f = synth_factors()
    res = ff_attribution(synth_pnl(f), factors=f)
    assert res["model"] == "ff5_mom"
    assert res["n_days"] == 2000 and res["n_dropped"] == 0
    assert res["hac_lags"] == DEFAULT_HAC_LAGS == 32
    assert res["loadings"]["mkt_rf"] == pytest.approx(BETA_MKT, abs=0.05)
    assert res["loadings"]["hml"] == pytest.approx(BETA_HML, abs=0.05)
    for c in ("smb", "rmw", "cma", "mom"):  # unplanted factors stay near zero
        assert abs(res["loadings"][c]) < 0.05
    assert abs(res["alpha"] - ALPHA) < 2 * res["alpha_se_hac"]
    assert 0.5 < res["r2"] < 0.995
    assert res["alpha_ann"] == pytest.approx(res["alpha"] * 252)
    assert res["ir_resid"] == pytest.approx(res["alpha_ann"] / res["resid_vol_ann"])
    assert set(res["loadings"]) == set(MODELS["ff5_mom"])


def test_alpha_t_responds_to_noise_scale():
    f = synth_factors()
    quiet = ff_attribution(synth_pnl(f, noise=0.001), factors=f)
    loud = ff_attribution(synth_pnl(f, noise=0.004), factors=f)
    assert quiet["alpha_t_hac"] > loud["alpha_t_hac"]
    assert quiet["alpha_t_hac"] > 2.0  # 3bps/day on 10bps noise over 2000d
    assert quiet["resid_vol_ann"] < loud["resid_vol_ann"]


def test_lags_override_is_used():
    f = synth_factors()
    res = ff_attribution(synth_pnl(f), factors=f, lags=5)
    assert res["hac_lags"] == 5
    # iid noise: the point estimate must not move with the lag choice
    assert res["alpha"] == pytest.approx(
        ff_attribution(synth_pnl(f), factors=f)["alpha"])


# --------------------------------------------------------------------------- #
# CAPM mode
# --------------------------------------------------------------------------- #

def test_capm_mode_returns_only_mkt_loading():
    f = synth_factors()
    res = ff_attribution(synth_pnl(f), factors=f, model="capm")
    assert res["model"] == "capm"
    assert set(res["loadings"]) == {"mkt_rf"}
    assert set(res["loadings_t_hac"]) == {"mkt_rf"}
    assert res["loadings"]["mkt_rf"] == pytest.approx(BETA_MKT, abs=0.05)


def test_unknown_model_raises():
    f = synth_factors()
    with pytest.raises(ValueError, match="unknown model"):
        ff_attribution(synth_pnl(f), factors=f, model="ff3")


# --------------------------------------------------------------------------- #
# Date alignment
# --------------------------------------------------------------------------- #

def test_inner_join_drops_extra_pnl_dates():
    f = synth_factors()
    pnl = synth_pnl(f)
    extra = pd.Series(0.123, index=pd.date_range("2030-01-01", periods=60))
    res = ff_attribution(pd.concat([pnl, extra]), factors=f)
    base = ff_attribution(pnl, factors=f)
    assert res["n_days"] == 2000
    assert res["n_pnl_days"] == 2060 and res["n_dropped"] == 60
    assert res["loadings"]["mkt_rf"] == pytest.approx(base["loadings"]["mkt_rf"])
    assert res["alpha"] == pytest.approx(base["alpha"])
    assert res["alpha_t_hac"] == pytest.approx(base["alpha_t_hac"])


def test_duplicate_pnl_dates_raise():
    f = synth_factors()
    pnl = synth_pnl(f)
    with pytest.raises(ValueError, match="duplicate"):
        ff_attribution(pd.concat([pnl, pnl.iloc[:3]]), factors=f)
