"""HMM regime detection — identify market regimes for position sizing.

Fits a 3-state Gaussian HMM on market features (returns, volatility, volume)
to classify each trading day into one of:
  - State 0: LOW-VOL TRENDING  (bull market, momentum works, short leg hurts)
  - State 1: MEAN-REVERTING    (range-bound, long-short signals work best)
  - State 2: CRISIS            (high vol, correlations spike, reduce exposure)

States are labelled post-hoc by sorting on volatility: lowest-vol state = trending,
mid-vol = mean-reverting, highest-vol = crisis.

The regime filter tells the backtest engine:
  - TRENDING:      reduce short leg to 25% weight (momentum dominates)
  - MEAN-REVERTING: full exposure (signal has most edge here)
  - CRISIS:        reduce all exposure to 25% (risk management)

Physics analogy: HMM is hidden state estimation — analogous to inferring
thermodynamic phases from observable quantities. Kalman filters estimate
continuous hidden states; HMMs estimate discrete hidden states.

Usage:
    python -m alpha_graph.signals.regime [--plot]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger

from alpha_graph.config import CACHE_DIR


# Regime labels assigned by volatility ranking
REGIME_LABELS = {0: "TRENDING", 1: "MEAN_REVERTING", 2: "CRISIS"}

# Exposure multipliers per regime
REGIME_EXPOSURE = {
    "TRENDING": {"long": 1.0, "short": 0.25},      # reduce shorts in bull
    "MEAN_REVERTING": {"long": 1.0, "short": 1.0},  # full exposure
    "CRISIS": {"long": 0.25, "short": 0.25},         # reduce everything
}


def prepare_features(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Compute HMM input features from SPY or market-wide data.

    Features (all computed on trailing windows to avoid lookahead):
      - ret_5d: 5-day rolling return
      - vol_21d: 21-day rolling volatility (annualized)
      - vol_ratio: ratio of 5-day vol to 21-day vol (vol regime change speed)
    """
    # Use equal-weighted market return if multiple tickers, else single ticker
    if "ticker" in prices.columns:
        daily = (
            prices.groupby("date")["close"]
            .mean()
            .reset_index()
            .sort_values("date")
            .set_index("date")
        )
        daily_ret = daily["close"].pct_change()
    else:
        prices = prices.sort_values("date").set_index("date")
        daily_ret = prices["close"].pct_change()

    features = pd.DataFrame(index=daily_ret.index)
    features["ret_5d"] = daily_ret.rolling(5).sum()
    features["vol_21d"] = daily_ret.rolling(lookback).std() * np.sqrt(252)
    features["vol_5d"] = daily_ret.rolling(5).std() * np.sqrt(252)
    features["vol_ratio"] = features["vol_5d"] / features["vol_21d"]

    features = features.dropna()
    return features


def fit_hmm(
    features: pd.DataFrame,
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
) -> tuple[GaussianHMM, np.ndarray, np.ndarray]:
    """Fit a Gaussian HMM on market features.

    Returns:
        model: fitted GaussianHMM
        states: array of state assignments per date
        state_probs: array of state probabilities per date (n_dates x n_states)
    """
    X = features[["ret_5d", "vol_21d", "vol_ratio"]].values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)

    states = model.predict(X)
    state_probs = model.predict_proba(X)

    logger.info(
        f"HMM fitted: {n_states} states, "
        f"log-likelihood={model.score(X):.1f}, "
        f"converged={model.monitor_.converged}"
    )

    return model, states, state_probs


def label_states(features: pd.DataFrame, states: np.ndarray) -> dict[int, str]:
    """Label HMM states by volatility: lowest vol = TRENDING, highest = CRISIS."""
    features = features.copy()
    features["state"] = states

    # Compute mean volatility per state
    vol_by_state = features.groupby("state")["vol_21d"].mean().sort_values()

    # Map: lowest vol -> TRENDING, mid -> MEAN_REVERTING, highest -> CRISIS
    labels = {}
    for i, state_id in enumerate(vol_by_state.index):
        labels[state_id] = REGIME_LABELS[i]

    return labels


def detect_regimes(market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run full regime detection pipeline.

    Returns DataFrame with columns: date, regime, regime_prob,
    long_exposure, short_exposure.
    """
    if market is None:
        market_path = CACHE_DIR / "market_data.parquet"
        if not market_path.exists():
            logger.error("No market data. Run market download first.")
            return pd.DataFrame()
        market = pd.read_parquet(market_path)

    features = prepare_features(market)
    if len(features) < 60:
        logger.error(f"Only {len(features)} days of features — need at least 60")
        return pd.DataFrame()

    model, states, state_probs = fit_hmm(features)
    state_labels = label_states(features, states)

    # Build output
    regimes = pd.DataFrame(index=features.index)
    regimes["state_id"] = states
    regimes["regime"] = [state_labels[s] for s in states]

    # Add probability of being in current state
    regimes["regime_prob"] = [state_probs[i, s] for i, s in enumerate(states)]

    # Add exposure multipliers
    regimes["long_exposure"] = regimes["regime"].map(
        lambda r: REGIME_EXPOSURE[r]["long"]
    )
    regimes["short_exposure"] = regimes["regime"].map(
        lambda r: REGIME_EXPOSURE[r]["short"]
    )

    regimes = regimes.reset_index()

    # Summary
    counts = regimes["regime"].value_counts()
    for regime, count in counts.items():
        pct = count / len(regimes) * 100
        logger.info(f"  {regime}: {count} days ({pct:.1f}%)")

    # Cache
    out_path = CACHE_DIR / "regimes.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    regimes.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(regimes)} regime observations to {out_path}")

    return regimes


def main():
    parser = argparse.ArgumentParser(description="HMM regime detection")
    parser.add_argument("--plot", action="store_true", help="Save regime plot")
    args = parser.parse_args()

    regimes = detect_regimes()
    if regimes.empty:
        return

    print("\n--- Regime Summary ---")
    print(regimes.groupby("regime").agg(
        days=("regime", "count"),
        avg_prob=("regime_prob", "mean"),
        long_exp=("long_exposure", "first"),
        short_exp=("short_exposure", "first"),
    ).to_string())

    print("\n--- Last 10 Days ---")
    print(regimes[["date", "regime", "regime_prob"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
