"""Unit tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.backtest.engine import (
    BacktestConfig,
    build_long_short_portfolio,
    compute_performance_metrics,
    compute_portfolio_return,
)


def _make_signals(n: int = 20) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n)],
        "combined_score": np.linspace(1.0, -1.0, n),
        "combined_confidence": rng.uniform(0.3, 0.9, n),
        "recommendation": ["BUY"] * (n // 3) + ["HOLD"] * (n // 3) + ["SELL"] * (n - 2 * (n // 3)),
        "fwd_return_21d": rng.normal(0.01, 0.05, n),
    })


def test_build_portfolio_correct_sizes():
    config = BacktestConfig(top_n_long=5, top_n_short=5)
    signals = _make_signals(20)
    portfolio = build_long_short_portfolio(signals, config)

    assert len(portfolio[portfolio["position"] == "LONG"]) == 5
    assert len(portfolio[portfolio["position"] == "SHORT"]) == 5
    assert portfolio[portfolio["position"] == "LONG"]["weight"].sum() == pytest.approx(1.0)
    assert portfolio[portfolio["position"] == "SHORT"]["weight"].sum() == pytest.approx(-1.0)


def test_portfolio_return_includes_costs():
    config = BacktestConfig(top_n_long=5, top_n_short=5, cost_bps=10)
    signals = _make_signals(20)
    portfolio = build_long_short_portfolio(signals, config)
    result = compute_portfolio_return(portfolio, config)

    assert result["total_cost"] > 0
    assert result["net_return"] < result["gross_return"]


def test_performance_metrics_positive_returns():
    returns = pd.Series([0.02, 0.01, 0.03, -0.01, 0.02, 0.01, 0.04, -0.02, 0.03, 0.01, 0.02, 0.01])
    metrics = compute_performance_metrics(returns)

    assert metrics["sharpe"] > 0
    assert metrics["total_return"] > 0
    assert metrics["max_drawdown"] <= 0
    assert 0 < metrics["deflated_sharpe_pvalue"] < 1


def test_performance_metrics_empty():
    metrics = compute_performance_metrics(pd.Series(dtype=float))
    assert metrics["sharpe"] == 0.0
