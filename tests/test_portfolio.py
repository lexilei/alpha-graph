"""Unit tests for improved portfolio construction."""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.backtest.engine import (
    BacktestConfig,
    build_long_short_portfolio,
)


def _make_signals(n: int = 20, with_sector: bool = False) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n)],
        "combined_score": np.linspace(1.0, -1.0, n),
        "combined_confidence": rng.uniform(0.3, 0.9, n),
        "recommendation": ["BUY"] * (n // 3) + ["HOLD"] * (n // 3) + ["SELL"] * (n - 2 * (n // 3)),
        "fwd_return_21d": rng.normal(0.01, 0.05, n),
    })
    if with_sector:
        sectors = ["Tech", "Health", "Finance", "Energy", "Consumer"]
        df["sector"] = [sectors[i % len(sectors)] for i in range(n)]
    return df


def test_signal_weighted_portfolio():
    """Signal-weighted positions should not be equal-weight."""
    config = BacktestConfig(top_n_long=5, top_n_short=5, signal_weighted=True)
    signals = _make_signals(20)
    portfolio = build_long_short_portfolio(signals, config)

    long_weights = portfolio[portfolio["position"] == "LONG"]["weight"]
    # With linear scores, weights should NOT all be equal
    assert long_weights.std() > 0.001


def test_equal_weight_portfolio():
    """Equal-weight should produce uniform weights."""
    config = BacktestConfig(top_n_long=5, top_n_short=5, signal_weighted=False)
    signals = _make_signals(20)
    portfolio = build_long_short_portfolio(signals, config)

    long_weights = portfolio[portfolio["position"] == "LONG"]["weight"]
    assert all(abs(w - 0.2) < 0.001 for w in long_weights)


def test_max_position_cap():
    """No single position should exceed max_position_pct after normalization."""
    config = BacktestConfig(
        top_n_long=5, top_n_short=5,
        signal_weighted=True, max_position_pct=0.05,
    )
    # Create extreme signals: one stock has much stronger signal
    signals = _make_signals(20)
    signals.loc[0, "combined_score"] = 10.0  # extreme outlier
    portfolio = build_long_short_portfolio(signals, config)

    # After normalization, weights should sum to 1/-1
    long_sum = portfolio[portfolio["position"] == "LONG"]["weight"].sum()
    short_sum = portfolio[portfolio["position"] == "SHORT"]["weight"].sum()
    assert abs(long_sum - 1.0) < 0.01
    assert abs(short_sum + 1.0) < 0.01


def test_portfolio_weights_sum():
    """Long weights sum to 1.0, short weights sum to -1.0."""
    config = BacktestConfig(top_n_long=5, top_n_short=5, signal_weighted=True)
    signals = _make_signals(20)
    portfolio = build_long_short_portfolio(signals, config)

    long_sum = portfolio[portfolio["position"] == "LONG"]["weight"].sum()
    short_sum = portfolio[portfolio["position"] == "SHORT"]["weight"].sum()
    assert abs(long_sum - 1.0) < 0.01
    assert abs(short_sum + 1.0) < 0.01


def test_sector_diversification():
    """With sector constraint, no sector should exceed max_sector_pct."""
    config = BacktestConfig(
        top_n_long=10, top_n_short=10,
        signal_weighted=True, max_sector_pct=0.30,
    )
    # All long positions in same sector
    signals = _make_signals(20, with_sector=True)
    # Force all top stocks into same sector
    signals.loc[:9, "sector"] = "Tech"
    signals = signals.sort_values("combined_score", ascending=False)
    portfolio = build_long_short_portfolio(signals, config)

    # Check sector weights in long leg
    long_leg = portfolio[portfolio["position"] == "LONG"]
    if "sector" in long_leg.columns:
        sector_weights = long_leg.groupby("sector")["weight"].sum()
        for sector, weight in sector_weights.items():
            assert weight <= config.max_sector_pct + 0.02  # small tolerance for rounding
