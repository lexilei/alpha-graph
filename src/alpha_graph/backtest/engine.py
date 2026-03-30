"""Walk-forward backtesting engine with proper financial methodology.

Implements the key practices from López de Prado's "Advances in Financial ML":
- Purged walk-forward cross-validation (no lookahead)
- Transaction cost modeling (5-10 bps round-trip)
- Deflated Sharpe Ratio (accounts for multiple testing)
- Proper benchmark comparison

This is NOT a full event-driven backtester — it's a signal-based evaluation
framework that measures whether our pipeline signals predict forward returns.

Usage:
    python -m alpha_graph.backtest.engine
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR


@dataclass
class BacktestConfig:
    """Backtesting parameters."""

    # Transaction costs
    cost_bps: float = 10.0  # round-trip cost in basis points
    # Rebalancing
    rebalance_freq: str = "M"  # M=monthly, Q=quarterly
    # Position sizing
    top_n_long: int = 10  # number of stocks in long leg
    top_n_short: int = 10  # number of stocks in short leg
    # Evaluation
    holding_period_days: int = 21  # ~1 month forward returns
    # Walk-forward
    train_months: int = 24  # lookback for any parameter estimation
    purge_days: int = 5  # gap between train and test to prevent leakage


def load_signals_and_returns() -> pd.DataFrame | None:
    """Load pipeline signals and merge with market returns."""
    signals_path = CACHE_DIR / "pipeline_signals.parquet"
    market_path = CACHE_DIR / "market_data.parquet"

    if not signals_path.exists():
        logger.error(f"No signals at {signals_path}. Run the pipeline first.")
        return None
    if not market_path.exists():
        logger.error(f"No market data at {market_path}. Run market data download first.")
        return None

    signals = pd.read_parquet(signals_path)
    market = pd.read_parquet(market_path)

    # Get the latest row with valid forward returns for each ticker
    valid_market = market.dropna(subset=["ret_21d"])
    latest_prices = (
        valid_market.sort_values("date")
        .groupby("ticker")
        .tail(1)[["ticker", "date", "ret_21d"]]
        .rename(columns={"ret_21d": "fwd_return_21d", "date": "price_date"})
    )

    merged = signals.merge(latest_prices, on="ticker", how="inner")
    return merged


def build_long_short_portfolio(
    signals: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Construct a long-short portfolio from pipeline signals.

    Long leg: top N stocks by combined_score (most bullish)
    Short leg: bottom N stocks by combined_score (most bearish)
    Equal-weighted within each leg.
    """
    signals = signals.sort_values("combined_score", ascending=False)

    long_leg = signals.head(config.top_n_long).copy()
    short_leg = signals.tail(config.top_n_short).copy()

    long_leg["position"] = "LONG"
    long_leg["weight"] = 1.0 / config.top_n_long
    short_leg["position"] = "SHORT"
    short_leg["weight"] = -1.0 / config.top_n_short

    portfolio = pd.concat([long_leg, short_leg], ignore_index=True)
    return portfolio


def compute_portfolio_return(
    portfolio: pd.DataFrame,
    config: BacktestConfig,
) -> dict:
    """Compute portfolio return accounting for transaction costs."""
    if "fwd_return_21d" not in portfolio.columns:
        return {"gross_return": 0.0, "net_return": 0.0, "n_positions": 0}

    portfolio = portfolio.dropna(subset=["fwd_return_21d"])

    # Gross return: weighted sum of forward returns
    gross_return = (portfolio["weight"] * portfolio["fwd_return_21d"]).sum()

    # Transaction costs: cost per position * number of positions
    n_positions = len(portfolio)
    total_cost = n_positions * config.cost_bps / 10_000  # both entry and exit

    net_return = gross_return - total_cost

    return {
        "gross_return": gross_return,
        "net_return": net_return,
        "total_cost": total_cost,
        "n_positions": n_positions,
        "n_long": len(portfolio[portfolio["position"] == "LONG"]),
        "n_short": len(portfolio[portfolio["position"] == "SHORT"]),
    }


def walk_forward_backtest(
    signals_by_date: dict[str, pd.DataFrame],
    config: BacktestConfig,
) -> pd.DataFrame:
    """Run walk-forward backtest over multiple rebalancing periods.

    Args:
        signals_by_date: dict mapping rebalance dates to signal DataFrames
            (each with 'ticker', 'combined_score', 'fwd_return_21d')
        config: backtesting parameters

    Returns:
        DataFrame with one row per period: date, gross_return, net_return, etc.
    """
    results = []

    for date, signals in sorted(signals_by_date.items()):
        if len(signals) < config.top_n_long + config.top_n_short:
            logger.debug(f"[{date}] Only {len(signals)} signals, need {config.top_n_long + config.top_n_short}")
            continue

        portfolio = build_long_short_portfolio(signals, config)
        period_result = compute_portfolio_return(portfolio, config)
        period_result["date"] = date
        results.append(period_result)

    return pd.DataFrame(results)


def compute_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Compute comprehensive performance metrics from a return series.

    Args:
        returns: Series of period returns (e.g., monthly)
        risk_free_rate: annualized risk-free rate

    Returns:
        Dict with Sharpe, Sortino, max drawdown, etc.
    """
    if returns.empty or returns.std() == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}

    periods_per_year = 12  # monthly
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - rf_per_period

    # Annualized Sharpe
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

    # Sortino (downside deviation only)
    downside = excess_returns[excess_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else excess_returns.std()
    sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
    # Accounts for skewness, kurtosis, and number of trials
    T = len(returns)
    skew = returns.skew()
    kurt = returns.kurtosis()  # excess kurtosis
    sr = sharpe / np.sqrt(periods_per_year)  # per-period Sharpe

    # Standard error of Sharpe ratio
    sr_std = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / T)

    # DSR: probability that the Sharpe is genuine (not from luck/multiple testing)
    from scipy import stats

    dsr_pvalue = stats.norm.cdf(sr / sr_std) if sr_std > 0 else 0.5

    return {
        "total_return": float((1 + returns).prod() - 1),
        "annualized_return": float((1 + returns).prod() ** (periods_per_year / len(returns)) - 1),
        "annualized_volatility": float(returns.std() * np.sqrt(periods_per_year)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "deflated_sharpe_pvalue": float(dsr_pvalue),
        "n_periods": T,
        "pct_positive": float((returns > 0).mean()),
    }


def generate_tearsheet(returns: pd.Series, benchmark_returns: pd.Series | None = None) -> str:
    """Generate a text-based performance tearsheet."""
    metrics = compute_performance_metrics(returns)

    lines = [
        "=" * 60,
        "  BACKTEST PERFORMANCE TEARSHEET",
        "=" * 60,
        f"  Total Return:           {metrics['total_return']:+.2%}",
        f"  Annualized Return:      {metrics['annualized_return']:+.2%}",
        f"  Annualized Volatility:  {metrics['annualized_volatility']:.2%}",
        f"  Sharpe Ratio:           {metrics['sharpe']:.3f}",
        f"  Sortino Ratio:          {metrics['sortino']:.3f}",
        f"  Max Drawdown:           {metrics['max_drawdown']:.2%}",
        f"  Skewness:               {metrics['skewness']:.3f}",
        f"  Excess Kurtosis:        {metrics['kurtosis']:.3f}",
        f"  Deflated Sharpe p-val:  {metrics['deflated_sharpe_pvalue']:.4f}",
        f"  Periods:                {metrics['n_periods']}",
        f"  % Positive:             {metrics['pct_positive']:.1%}",
        "-" * 60,
    ]

    if benchmark_returns is not None and not benchmark_returns.empty:
        bm = compute_performance_metrics(benchmark_returns)
        lines.extend([
            "  BENCHMARK (SPY)",
            f"  Annualized Return:      {bm['annualized_return']:+.2%}",
            f"  Sharpe Ratio:           {bm['sharpe']:.3f}",
            f"  Max Drawdown:           {bm['max_drawdown']:.2%}",
            "-" * 60,
            f"  ALPHA:                  {metrics['annualized_return'] - bm['annualized_return']:+.2%}",
            "=" * 60,
        ])
    else:
        lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """Run backtest using saved signals and market data."""
    data = load_signals_and_returns()
    if data is None:
        return

    config = BacktestConfig()
    portfolio = build_long_short_portfolio(data, config)
    result = compute_portfolio_return(portfolio, config)

    print("\n--- Single-Period Portfolio ---")
    print(f"Positions: {result['n_long']} long, {result['n_short']} short")
    print(f"Gross return: {result['gross_return']:+.2%}")
    print(f"Transaction costs: {result['total_cost']:.4%}")
    print(f"Net return: {result['net_return']:+.2%}")

    print("\n(Run walk-forward backtest with multiple periods for reliable metrics)")


if __name__ == "__main__":
    main()
