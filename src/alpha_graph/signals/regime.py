"""HMM regime detection — identify market regimes for position sizing.

Fits a Gaussian HMM on market features (returns, volatility, volume, breadth)
to classify each trading day into one of:
  - State 0: LOW-VOL TRENDING  (bull market, momentum works, short leg hurts)
  - State 1: MEAN-REVERTING    (range-bound, long-short signals work best)
  - State 2: CRISIS            (high vol, correlations spike, reduce exposure)

Improvements over v1:
  - StandardScaler normalization (features were on different scales)
  - Market breadth feature: % of stocks above 50-day SMA (participation metric)
  - BIC-based model selection: tests n_states in {3, 4} and picks the best

States are labelled post-hoc by sorting on volatility: lowest-vol state = trending,
mid-vol = mean-reverting, highest-vol = crisis.

The regime filter tells the backtest engine:
  - TRENDING:      reduce short leg to 25% weight (momentum dominates)
  - MEAN-REVERTING: full exposure (signal has most edge here)
  - CRISIS:        reduce all exposure to 25% (risk management)

Usage:
    python -m alpha_graph.signals.regime [--plot]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger
from sklearn.preprocessing import StandardScaler

from alpha_graph.config import CACHE_DIR


# Regime labels assigned by volatility ranking
REGIME_LABELS = {0: "TRENDING", 1: "MEAN_REVERTING", 2: "CRISIS"}

# Extended labels for 4-state model
REGIME_LABELS_4 = {0: "TRENDING", 1: "LOW_VOL", 2: "MEAN_REVERTING", 3: "CRISIS"}

# Exposure multipliers per regime
REGIME_EXPOSURE = {
    "TRENDING": {"long": 1.0, "short": 0.25},      # reduce shorts in bull
    "LOW_VOL": {"long": 1.0, "short": 0.5},         # moderate short exposure
    "MEAN_REVERTING": {"long": 1.0, "short": 1.0},  # full exposure
    "CRISIS": {"long": 0.25, "short": 0.25},         # reduce everything
}


def prepare_features(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Compute HMM input features from market-wide data.

    Features (all computed on trailing windows to avoid lookahead):
      - ret_5d: 5-day rolling return
      - vol_21d: 21-day rolling volatility (annualized)
      - vol_ratio: ratio of 5-day vol to 21-day vol (vol regime change speed)
      - breadth: % of stocks above their 50-day SMA (market participation)
    """
    prices_copy = prices.copy()

    # Compute market breadth if multiple tickers available
    breadth_series = None
    if "ticker" in prices_copy.columns and prices_copy["ticker"].nunique() > 1:
        # For each ticker-date, check if close > 50-day SMA
        prices_copy = prices_copy.sort_values(["ticker", "date"])
        prices_copy["sma_50"] = prices_copy.groupby("ticker")["close"].transform(
            lambda s: s.rolling(50, min_periods=30).mean()
        )
        prices_copy["above_sma50"] = (prices_copy["close"] > prices_copy["sma_50"]).astype(float)

        # Breadth = % of stocks above 50-day SMA each day
        breadth_series = (
            prices_copy.groupby("date")["above_sma50"]
            .mean()
        )
        breadth_series.name = "breadth"

    # Compute market-level return series
    if "ticker" in prices_copy.columns:
        daily = (
            prices_copy.groupby("date")["close"]
            .mean()
            .reset_index()
            .sort_values("date")
            .set_index("date")
        )
        daily_ret = daily["close"].pct_change()
    else:
        prices_copy = prices_copy.sort_values("date").set_index("date")
        daily_ret = prices_copy["close"].pct_change()

    features = pd.DataFrame(index=daily_ret.index)
    features["ret_5d"] = daily_ret.rolling(5).sum()
    features["vol_21d"] = daily_ret.rolling(lookback).std() * np.sqrt(252)
    features["vol_5d"] = daily_ret.rolling(5).std() * np.sqrt(252)
    features["vol_ratio"] = features["vol_5d"] / features["vol_21d"]

    # Add breadth feature if available
    if breadth_series is not None:
        features = features.join(breadth_series, how="left")
    else:
        features["breadth"] = 0.5  # default to 50% when single-ticker data

    features = features.dropna()
    return features


def fit_hmm(
    features: pd.DataFrame,
    n_states: int | None = None,
    n_iter: int = 200,
    random_state: int = 42,
) -> tuple[GaussianHMM, np.ndarray, np.ndarray, StandardScaler, int]:
    """Fit a Gaussian HMM on market features with proper scaling.

    If n_states is None, uses BIC to select between 3 and 4 states.
    Features are standardized before fitting (StandardScaler) since
    returns, volatility, and breadth are on different scales.

    Returns:
        model: fitted GaussianHMM
        states: array of state assignments per date
        state_probs: array of state probabilities per date (n_dates x n_states)
        scaler: fitted StandardScaler (needed for live prediction)
        best_n: number of states selected
    """
    feature_cols = ["ret_5d", "vol_21d", "vol_ratio"]
    if "breadth" in features.columns:
        feature_cols.append("breadth")

    X_raw = features[feature_cols].values

    # Standardize features — critical because vol_21d is ~0.1-0.5 annualized
    # while ret_5d is ~-0.1 to +0.1 and breadth is 0-1
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Model selection via BIC if n_states not specified
    if n_states is None:
        candidates = [3, 4]
        best_bic = np.inf
        best_n = 3

        for n in candidates:
            try:
                model_candidate = GaussianHMM(
                    n_components=n,
                    covariance_type="full",
                    n_iter=n_iter,
                    random_state=random_state,
                )
                model_candidate.fit(X)

                # BIC = -2 * log_likelihood + k * log(n_samples)
                # where k = number of free parameters
                log_ll = model_candidate.score(X) * len(X)
                n_features = X.shape[1]
                # Parameters: transition matrix (n*n-n) + means (n*d) + covariances (n*d*(d+1)/2) + initial probs (n-1)
                k = (n * n - n) + (n * n_features) + (n * n_features * (n_features + 1) // 2) + (n - 1)
                bic = -2 * log_ll + k * np.log(len(X))

                logger.info(f"  HMM n_states={n}: BIC={bic:.1f}, log_ll={log_ll:.1f}")

                if bic < best_bic:
                    best_bic = bic
                    best_n = n
            except Exception as e:
                logger.warning(f"  HMM n_states={n} failed: {e}")

        n_states = best_n
        logger.info(f"BIC selected n_states={n_states}")

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

    return model, states, state_probs, scaler, n_states


def label_states(
    features: pd.DataFrame,
    states: np.ndarray,
    n_states: int = 3,
) -> dict[int, str]:
    """Label HMM states by volatility: lowest vol = TRENDING, highest = CRISIS.

    Handles both 3-state and 4-state models:
    - 3 states: TRENDING, MEAN_REVERTING, CRISIS
    - 4 states: TRENDING, LOW_VOL, MEAN_REVERTING, CRISIS
    """
    features = features.copy()
    features["state"] = states

    # Compute mean volatility per state
    vol_by_state = features.groupby("state")["vol_21d"].mean().sort_values()

    # Select label mapping based on number of states
    if n_states == 4:
        label_map = REGIME_LABELS_4
    else:
        label_map = REGIME_LABELS

    labels = {}
    for i, state_id in enumerate(vol_by_state.index):
        if i < len(label_map):
            labels[state_id] = label_map[i]
        else:
            labels[state_id] = f"STATE_{i}"

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

    model, states, state_probs, scaler, best_n = fit_hmm(features)
    state_labels = label_states(features, states, n_states=best_n)

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
