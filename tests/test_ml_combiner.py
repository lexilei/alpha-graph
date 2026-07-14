"""Unit tests for the ML Signal Combiner module."""

import numpy as np
import pandas as pd
import pytest

import alpha_graph.signals.ml_combiner as ml_combiner
from alpha_graph.signals.ml_combiner import (
    FEATURE_COLS,
    LGBM_PARAMS,
    compute_signal_metrics,
)

try:
    import lightgbm
    HAS_LGB = True
except ImportError:
    lightgbm = None
    HAS_LGB = False


def _make_results(n: int = 200, ic: float = 0.1) -> pd.DataFrame:
    """Create synthetic walk-forward results with controllable IC."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-01", periods=n // 10, freq="D").repeat(10)[:n]
    actual = rng.normal(0.01, 0.05, n)
    # predicted = actual * correlation + noise
    noise = rng.normal(0, 0.05, n)
    predicted = actual * ic * 10 + noise * (1 - abs(ic))
    return pd.DataFrame({
        "date": dates,
        "ticker": [f"T{i % 10:02d}" for i in range(n)],
        "predicted_return": predicted,
        "actual_return": actual,
    })


def test_compute_signal_metrics_returns_expected_keys():
    results = _make_results(200)
    metrics = compute_signal_metrics(results)
    expected_keys = {"overall_ic", "mean_monthly_ic", "icir", "n_months",
                     "pct_positive_ic", "long_short_spread"}
    assert expected_keys.issubset(set(metrics.keys()))


def test_compute_signal_metrics_positive_ic():
    """With positively correlated predictions, IC should be positive."""
    rng = np.random.RandomState(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n // 10, freq="D").repeat(10)[:n]
    actual = rng.normal(0.01, 0.05, n)
    predicted = actual + rng.normal(0, 0.01, n)  # strong correlation
    results = pd.DataFrame({
        "date": dates,
        "ticker": [f"T{i % 10:02d}" for i in range(n)],
        "predicted_return": predicted,
        "actual_return": actual,
    })
    metrics = compute_signal_metrics(results)
    assert metrics["overall_ic"] > 0.3  # should be strongly positive


def test_compute_signal_metrics_empty():
    metrics = compute_signal_metrics(pd.DataFrame())
    assert metrics == {}


def test_lgbm_params_conservative():
    """Ensure hyperparameters are conservative to prevent overfitting."""
    assert LGBM_PARAMS["num_leaves"] <= 31  # shallow trees
    assert LGBM_PARAMS["learning_rate"] <= 0.1
    assert LGBM_PARAMS["reg_alpha"] > 0  # L1 regularization active
    assert LGBM_PARAMS["reg_lambda"] > 0  # L2 regularization active
    assert LGBM_PARAMS["min_child_samples"] >= 5


def test_feature_cols_expected():
    """Ensure all expected features are defined."""
    expected = {
        "cosine_similarity", "momentum_21d", "momentum_5d",
        "volatility_21d", "volume_zscore",
    }
    assert expected.issubset(set(FEATURE_COLS))


# --- Model persistence (native LightGBM format, no pickle) ---

def _tiny_fit(n: int = 200):
    """Fit a small LGBMRegressor on synthetic data."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = X["momentum_21d"] * 0.5 - X["volatility_21d"] * 0.3 + rng.normal(0, 0.1, n)
    params = dict(LGBM_PARAMS)
    params["n_estimators"] = 20
    model = lightgbm.LGBMRegressor(**params)
    model.fit(X, y)
    return model, X


@pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
def test_native_save_load_round_trip_identical_predictions(tmp_path, monkeypatch):
    """Native text save/load must reproduce LGBMRegressor predictions exactly."""
    monkeypatch.setattr(ml_combiner, "CACHE_DIR", tmp_path)
    model, X = _tiny_fit()

    path = ml_combiner._save_model(model)
    assert path == tmp_path / ml_combiner.MODEL_FILENAME_NATIVE
    assert path.suffix == ".txt"
    # Native text format, not a pickle
    assert path.read_bytes()[:4] != b"\x80\x04"
    assert b"tree" in path.read_bytes()[:100]

    loaded = ml_combiner._load_model()
    assert isinstance(loaded, lightgbm.Booster)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


@pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
def test_stale_legacy_pickle_ignored(tmp_path, monkeypatch):
    """An old ml_combiner_model.pkl must be ignored, not loaded."""
    monkeypatch.setattr(ml_combiner, "CACHE_DIR", tmp_path)
    (tmp_path / ml_combiner.MODEL_FILENAME_LEGACY).write_bytes(b"\x80\x04 old pickle")
    assert ml_combiner._load_model() is None


def test_load_model_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_combiner, "CACHE_DIR", tmp_path)
    assert ml_combiner._load_model() is None
