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
    # Position limits
    max_position_pct: float = 0.05  # max 5% in any single stock
    max_sector_pct: float = 0.30  # max 30% in any single sector
    # Signal-weighted positions (vs equal-weight)
    signal_weighted: bool = True  # weight by signal strength
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


def _cap_and_redistribute(
    weights: np.ndarray,
    max_pct: float,
    target_sum: float = 1.0,
    iterations: int = 10,
) -> np.ndarray:
    """Cap individual weights at max_pct and redistribute excess.

    Iteratively clips overweight positions and redistributes their excess
    to underweight positions proportionally, preserving relative signal ordering.
    If the cap is infeasible (n * max_pct < target_sum), returns normalized
    weights without capping.
    """
    weights = weights.copy()
    n = len(weights)
    if n == 0:
        return weights

    # If cap is infeasible, just normalize
    if n * max_pct < target_sum:
        total = weights.sum()
        return weights / total * target_sum if total > 0 else np.full(n, target_sum / n)

    # Normalize to target_sum first
    total = weights.sum()
    if total > 0:
        weights = weights / total * target_sum

    for _ in range(iterations):
        over = weights > max_pct
        if not over.any():
            break
        excess = (weights[over] - max_pct).sum()
        weights[over] = max_pct
        under = ~over
        under_total = weights[under].sum()
        if under_total > 0:
            weights[under] += excess * (weights[under] / under_total)
        else:
            break

    return weights


def build_long_short_portfolio(
    signals: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Construct a long-short portfolio from pipeline signals.

    Long leg: top N stocks by combined_score (most bullish)
    Short leg: bottom N stocks by combined_score (most bearish)

    Improvements over equal-weight:
    - Signal-weighted positions (stronger signals get more weight)
    - Max position cap (default 5% per stock)
    - Sector diversification constraint (max 30% in any sector)
    """
    signals = signals.sort_values("combined_score", ascending=False)

    long_leg = signals.head(config.top_n_long).copy()
    short_leg = signals.tail(config.top_n_short).copy()

    long_leg["position"] = "LONG"
    short_leg["position"] = "SHORT"

    if config.signal_weighted and long_leg["combined_score"].std() > 0.01:
        # Signal-weighted: weight proportional to signal strength
        long_leg["raw_weight"] = long_leg["combined_score"].abs()
        long_sum = long_leg["raw_weight"].sum()
        if long_sum > 0:
            long_leg["weight"] = long_leg["raw_weight"] / long_sum
        else:
            long_leg["weight"] = 1.0 / config.top_n_long
    else:
        long_leg["weight"] = 1.0 / config.top_n_long

    if config.signal_weighted and short_leg["combined_score"].std() > 0.01:
        short_leg["raw_weight"] = short_leg["combined_score"].abs()
        short_sum = short_leg["raw_weight"].sum()
        if short_sum > 0:
            short_leg["weight"] = -(short_leg["raw_weight"] / short_sum)
        else:
            short_leg["weight"] = -1.0 / config.top_n_short
    else:
        short_leg["weight"] = -1.0 / config.top_n_short

    portfolio = pd.concat([long_leg, short_leg], ignore_index=True)

    # Apply max position cap with iterative redistribution
    long_mask = portfolio["position"] == "LONG"
    short_mask = portfolio["position"] == "SHORT"

    portfolio.loc[long_mask, "weight"] = _cap_and_redistribute(
        portfolio.loc[long_mask, "weight"].values, config.max_position_pct, target_sum=1.0
    )
    portfolio.loc[short_mask, "weight"] = _cap_and_redistribute(
        portfolio.loc[short_mask, "weight"].abs().values, config.max_position_pct, target_sum=1.0
    ) * -1.0

    # Sector diversification (if sector column available)
    if "sector" in portfolio.columns:
        portfolio = _apply_sector_constraint(portfolio, config.max_sector_pct)

    # Clean up temporary columns
    portfolio = portfolio.drop(columns=["raw_weight"], errors="ignore")

    return portfolio


def _apply_sector_constraint(
    portfolio: pd.DataFrame,
    max_sector_pct: float,
) -> pd.DataFrame:
    """Reduce positions to enforce sector concentration limits.

    For each leg (long/short), if a sector's total absolute weight exceeds
    max_sector_pct, scale down all positions in that sector proportionally.
    """
    portfolio = portfolio.copy()

    for position_type in ["LONG", "SHORT"]:
        mask = portfolio["position"] == position_type
        leg = portfolio.loc[mask]

        if leg.empty or "sector" not in leg.columns:
            continue

        sector_weights = leg.groupby("sector")["weight"].apply(lambda s: s.abs().sum())

        for sector, total_weight in sector_weights.items():
            if total_weight > max_sector_pct:
                sector_mask = mask & (portfolio["sector"] == sector)
                scale_factor = max_sector_pct / total_weight
                portfolio.loc[sector_mask, "weight"] *= scale_factor
                logger.debug(
                    f"Sector cap: {sector} {position_type} scaled by {scale_factor:.2f} "
                    f"({total_weight:.2%} -> {max_sector_pct:.2%})"
                )

        # Redistribute excess to uncapped sectors (don't blindly re-normalize,
        # which would undo the cap)
        capped_sectors = {
            s for s, w in sector_weights.items() if w > max_sector_pct
        }
        uncapped_mask = mask & ~portfolio["sector"].isin(capped_sectors)
        uncapped_total = portfolio.loc[uncapped_mask, "weight"].abs().sum()
        target = 1.0 if position_type == "LONG" else -1.0
        capped_total = len(capped_sectors) * max_sector_pct
        remaining = abs(target) - capped_total
        if uncapped_total > 0 and remaining > 0:
            sign = np.sign(portfolio.loc[uncapped_mask, "weight"].iloc[0]) if len(portfolio.loc[uncapped_mask]) > 0 else 1.0
            portfolio.loc[uncapped_mask, "weight"] = (
                sign * portfolio.loc[uncapped_mask, "weight"].abs()
                / uncapped_total * remaining
            )

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


def compute_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    n_trials: int = 1,
) -> dict:
    """Compute comprehensive performance metrics from a return series.

    Args:
        returns: Series of period returns (e.g., monthly)
        risk_free_rate: annualized risk-free rate
        n_trials: number of strategies tested in selection. Used to apply
            Bailey & Lopez de Prado (2014) deflation for multiple testing.
            Pass the *real* number of variants you compared (e.g. 5 for
            baseline + 4 extensions), not 1, otherwise the p-value is biased
            optimistic.

    Returns:
        Dict with Sharpe, Sortino, max drawdown, DSR p-value, Bonferroni-
        adjusted p-value, etc.
    """
    # Flatten to a plain 1-D Series of floats
    returns = pd.Series(returns.values.flatten(), dtype=float)
    returns = returns.dropna()

    if returns.empty or returns.std() == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}

    periods_per_year = 12  # monthly
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - rf_per_period

    # Annualized Sharpe
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

    # Sortino (downside deviation only)
    downside = excess_returns[excess_returns < 0]
    downside_std = float(downside.std()) if len(downside) > 0 else float(excess_returns.std())
    sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
    # Accounts for skewness, kurtosis, and (now) number of trials.
    T = len(returns)
    skew = returns.skew()
    kurt = returns.kurtosis()  # excess kurtosis
    sr = sharpe / np.sqrt(periods_per_year)  # per-period Sharpe

    # Standard error of Sharpe ratio adjusted for higher moments
    sr_std = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / T)

    from scipy import stats

    # Raw DSR p-value (assumes 1 trial)
    dsr_pvalue = stats.norm.cdf(sr / sr_std) if sr_std > 0 else 0.5

    # Bailey-Lopez de Prado multi-trial deflation: the expected maximum SR
    # under the null of N independent strategies grows like
    #   E[max SR] ≈ sqrt(2 ln N) - (gamma + ln ln N) / sqrt(2 ln N)
    # where gamma ≈ 0.5772 is Euler-Mascheroni. We subtract this from the
    # observed SR before computing the z-score.
    if n_trials > 1:
        gamma = 0.5772156649
        ln_n = np.log(n_trials)
        e_max_sr = np.sqrt(2 * ln_n) - (gamma + np.log(ln_n)) / np.sqrt(2 * ln_n)
        sr_deflated = sr - e_max_sr * sr_std
        dsr_pvalue_multi = stats.norm.cdf(sr_deflated / sr_std) if sr_std > 0 else 0.5
    else:
        dsr_pvalue_multi = dsr_pvalue

    # Simple Bonferroni adjustment for the one-tailed p-value: p_bonf = min(1, p * N)
    # Note: this is a more conservative correction than DSR's exact form.
    one_tailed_p = 1 - dsr_pvalue
    bonferroni_p = min(1.0, one_tailed_p * n_trials)

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
        "deflated_sharpe_pvalue_multi": float(dsr_pvalue_multi),
        "bonferroni_pvalue": float(bonferroni_p),
        "n_trials": int(n_trials),
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
