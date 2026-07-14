"""Tests for alpha_graph.portfolio — synthetic data only.

Covers construction (quantile L/S weights, cap redistribution), the cost
model, the share-based backtest engine, and the promotion-gate summary.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.portfolio import CostModel, cap_and_redistribute, quantile_ls_weights

TOL = 1e-12


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #

def test_quantile_weights_basic_and_nan_exclusion():
    scores = pd.Series(
        {f"N{i:02d}": float(i) for i in range(10)} | {"NAN1": np.nan}
    )
    w = quantile_ls_weights(scores, n_quantiles=5, direction=1, max_weight=0.5)
    # 10 valid names, 5 quantiles -> 2 per bin; top {N08,N09}, bottom {N00,N01}
    assert set(w.index) == {"N08", "N09", "N00", "N01"}
    assert w["N08"] == pytest.approx(0.5, abs=TOL)
    assert w["N09"] == pytest.approx(0.5, abs=TOL)
    assert w["N00"] == pytest.approx(-0.5, abs=TOL)
    assert w["N01"] == pytest.approx(-0.5, abs=TOL)
    assert abs(w.sum()) < TOL                      # dollar neutral
    assert abs(w.abs().sum() - 2.0) < TOL          # gross 2.0
    assert "NAN1" not in w.index


def test_quantile_weights_direction_flip():
    scores = pd.Series({f"N{i}": float(i) for i in range(8)})
    w_pos = quantile_ls_weights(scores, n_quantiles=4, direction=1, max_weight=0.5)
    w_neg = quantile_ls_weights(scores, n_quantiles=4, direction=-1, max_weight=0.5)
    assert ((w_neg + w_pos).abs() < TOL).all()     # exact sign flip


def test_quantile_weights_deterministic_ties():
    # Five names tied at the top score; with 10 names / 5 quantiles the top
    # bin takes 2 names — the tie must break alphabetically by ticker.
    scores = pd.Series(
        {"A": 9.0, "B": 9.0, "C": 9.0, "D": 9.0, "E": 9.0,
         "F": 1.0, "G": 2.0, "H": 3.0, "I": 4.0, "J": 5.0}
    )
    w1 = quantile_ls_weights(scores, n_quantiles=5, max_weight=0.5)
    w2 = quantile_ls_weights(scores.sample(frac=1.0, random_state=7), n_quantiles=5,
                             max_weight=0.5)
    # ascending sort: F,G,H,I,J then A,B,C,D,E -> top-2 = {D, E}; bottom-2 = {F, G}
    assert set(w1[w1 > 0].index) == {"D", "E"}
    assert set(w1[w1 < 0].index) == {"F", "G"}
    pd.testing.assert_series_equal(w1, w2)         # input order irrelevant


def test_quantile_weights_validation():
    scores = pd.Series({"A": 1.0, "B": 2.0, "C": np.nan})
    with pytest.raises(ValueError):                # 2 valid < 3 quantiles
        quantile_ls_weights(scores, n_quantiles=3)
    with pytest.raises(ValueError):
        quantile_ls_weights(scores, n_quantiles=2, direction=0)
    with pytest.raises(ValueError):                # infeasible cap: 1 name/leg at 0.05
        quantile_ls_weights(pd.Series({"A": 1.0, "B": 2.0}), n_quantiles=2,
                            max_weight=0.05)


def test_cap_and_redistribute():
    # one round: 0.5 capped at 0.4, excess pro-rata to the others
    w = cap_and_redistribute(pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}), cap=0.4)
    assert w["A"] == pytest.approx(0.4, abs=TOL)
    assert w["B"] == pytest.approx(0.36, abs=TOL)
    assert w["C"] == pytest.approx(0.24, abs=TOL)
    assert w.sum() == pytest.approx(1.0, abs=TOL)
    # two rounds: redistribution pushes B over the cap, second pass fixes it
    w2 = cap_and_redistribute(pd.Series({"A": 0.6, "B": 0.35, "C": 0.05}), cap=0.4)
    assert w2["A"] == pytest.approx(0.4, abs=TOL)
    assert w2["B"] == pytest.approx(0.4, abs=TOL)
    assert w2["C"] == pytest.approx(0.2, abs=TOL)
    assert w2.sum() == pytest.approx(1.0, abs=TOL)
    assert (w2 <= 0.4 + 1e-9).all()
    # infeasible
    with pytest.raises(ValueError):
        cap_and_redistribute(pd.Series({"A": 0.5, "B": 0.5}), cap=0.3)
    # exact fit is feasible
    w3 = cap_and_redistribute(pd.Series({"A": 0.7, "B": 0.3}), cap=0.5)
    assert w3["A"] == pytest.approx(0.5, abs=TOL)
    assert w3["B"] == pytest.approx(0.5, abs=TOL)


# --------------------------------------------------------------------------- #
# costs
# --------------------------------------------------------------------------- #

def test_cost_model_rates():
    cm = CostModel(half_spread_bps=10.0, commission_bps=5.0, borrow_bps_pa=252.0)
    assert cm.trading_cost_rate == pytest.approx(0.0015, abs=TOL)
    assert cm.daily_borrow_rate == pytest.approx(1e-4, abs=TOL)
    cm2 = CostModel(half_spread_bps=10.0, commission_bps=5.0, borrow_bps_pa=252.0,
                    doubled=True)
    assert cm2.trading_cost_rate == pytest.approx(0.0030, abs=TOL)
    assert cm2.daily_borrow_rate == pytest.approx(2e-4, abs=TOL)
    with pytest.raises(ValueError):
        CostModel(half_spread_bps=-1.0)
