"""Anti-Momentum signal generator for paper trading.

Production signal pipeline:
1. Refresh market data (yfinance) and 8-K events (SEC EDGAR)
2. Build feature panel (ML combiner features + anti-momentum features)
3. Train walk-forward LightGBM with expanded feature set
4. Generate current buy/sell signals for portfolio rebalance

The Anti-Momentum enhancement adds two mean-reversion features:
- dist_52w_high: distance from 52-week high (catches falling knives to avoid)
- reversal_5d: negative 5-day return (short-term reversal signal)

These features gave Sharpe 1.91, bilateral alpha (short-leg Sharpe +0.75),
and survived the Jan 2025 momentum crash better than all other variants.

Usage:
    python -m alpha_graph.trading.signal_generator [--refresh] [--top-n 10]
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR


# Anti-momentum feature columns added on top of base ML combiner
ANTI_MOMENTUM_FEATURES = ["signal", "dist_52w_high", "reversal_5d"]

# LightGBM hyperparameters (same as backtest — conservative regularization)
LGBM_PARAMS = {
    "n_estimators": 150,
    "num_leaves": 15,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 10,
    "verbose": -1,
}

MODEL_PATH = CACHE_DIR / "anti_momentum_model.pkl"
SIGNALS_PATH = CACHE_DIR / "anti_momentum_signals.parquet"


def refresh_market_data(max_tickers: int = 100, years_back: int = 3) -> pd.DataFrame:
    """Download latest market data via yfinance."""
    from alpha_graph.data.market import download_and_cache

    logger.info("Refreshing market data...")
    prices = download_and_cache(max_tickers=max_tickers, years_back=years_back)
    logger.info(f"Market data: {len(prices)} rows, {prices['ticker'].nunique()} tickers")
    return prices


def refresh_event_signals() -> pd.DataFrame:
    """Recompute 8-K event signals from local filings."""
    from alpha_graph.signals.event_signal import compute_all

    logger.info("Refreshing 8-K event signals...")
    events = compute_all()
    logger.info(f"Event signals: {len(events)} tickers")
    return events


def refresh_regime() -> pd.DataFrame:
    """Rerun HMM regime detection on latest market data."""
    from alpha_graph.signals.regime import detect_regimes

    logger.info("Refreshing regime detection...")
    regimes = detect_regimes()
    logger.info(f"Regime: {regimes.iloc[-1]['regime'] if not regimes.empty else 'UNKNOWN'}")
    return regimes


def build_anti_momentum_features(
    market: pd.DataFrame,
    ml_preds: pd.DataFrame,
) -> pd.DataFrame:
    """Add anti-momentum features to ML combiner predictions.

    Features:
    - dist_52w_high: (price / 252-day max) - 1  →  negative = far from high
    - reversal_5d: -(5-day return)  →  positive = recent loser (reversal candidate)
    """
    market = market.sort_values(["ticker", "date"]).copy()

    # 52-week high distance
    market["high_252d"] = market.groupby("ticker")["close"].transform(
        lambda s: s.rolling(252, min_periods=63).max()
    )
    market["dist_52w_high"] = market["close"] / market["high_252d"] - 1

    # Short-term reversal
    market["reversal_5d"] = -market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(5)
    )

    # Aggregate to monthly (last observation per ticker per month)
    market["month"] = market["date"].dt.to_period("M")
    new_feat = market.groupby(["ticker", "month"]).last()[
        ["dist_52w_high", "reversal_5d"]
    ].reset_index()

    # Merge into ML predictions
    ml_preds = ml_preds.copy()
    ml_preds["month"] = ml_preds["date"].dt.to_period("M")
    enhanced = ml_preds.merge(new_feat, on=["ticker", "month"], how="left")

    logger.info(
        f"Anti-momentum features: {enhanced['dist_52w_high'].notna().mean():.0%} coverage "
        f"(dist_52w_high), {enhanced['reversal_5d'].notna().mean():.0%} (reversal_5d)"
    )

    return enhanced


def train_anti_momentum_model(enhanced: pd.DataFrame, train_months: int = 12) -> object:
    """Walk-forward train and return the latest Anti-Momentum LightGBM model.

    Uses the same protocol as backtesting: rolling 12-month train window,
    retrain monthly, predict on OOS month.

    Returns the model trained on the most recent 12 months of data.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm required. Install: pip install lightgbm")
        return None

    months = sorted(enhanced["month"].unique())
    if len(months) < train_months + 1:
        logger.warning(
            f"Only {len(months)} months available, need {train_months + 1}. "
            f"Training on all available data."
        )
        train_data = enhanced.dropna(subset=ANTI_MOMENTUM_FEATURES + ["actual_return"])
        if len(train_data) < 100:
            logger.error("Not enough training data")
            return None

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            train_data[ANTI_MOMENTUM_FEATURES],
            train_data["actual_return"],
        )
        return model

    # Use the last train_months of data for the production model
    train_start = months[-train_months - 1]
    train_end = months[-1]
    train_data = enhanced[
        (enhanced["month"] >= train_start) & (enhanced["month"] <= train_end)
    ].dropna(subset=ANTI_MOMENTUM_FEATURES + ["actual_return"])

    if len(train_data) < 100:
        logger.error(f"Only {len(train_data)} training samples — need at least 100")
        return None

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        train_data[ANTI_MOMENTUM_FEATURES],
        train_data["actual_return"],
    )

    logger.info(
        f"Trained Anti-Momentum model on {len(train_data)} samples "
        f"({train_start}–{train_end})"
    )

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {MODEL_PATH}")

    return model


def generate_signals(top_n: int = 10, retrain: bool = True) -> pd.DataFrame:
    """Generate current Anti-Momentum buy/sell signals.

    Steps:
    1. Load ML combiner predictions and market data
    2. Compute anti-momentum features for latest cross-section
    3. Either retrain model or load saved model
    4. Predict and rank → top_n longs, bottom_n shorts

    Returns DataFrame with: ticker, signal, predicted_return, recommendation, confidence
    """
    # Load data
    preds_path = CACHE_DIR / "ml_combiner_predictions.parquet"
    market_path = CACHE_DIR / "market_data.parquet"

    if not preds_path.exists():
        logger.error(
            "No ML combiner predictions. Run: "
            "python -m alpha_graph.signals.ml_combiner --train"
        )
        return pd.DataFrame()
    if not market_path.exists():
        logger.error(
            "No market data. Run: "
            "python -m alpha_graph.data.market"
        )
        return pd.DataFrame()

    preds = pd.read_parquet(preds_path)
    market = pd.read_parquet(market_path)
    preds["date"] = pd.to_datetime(preds["date"])
    market["date"] = pd.to_datetime(market["date"])

    # Build anti-momentum features
    enhanced = build_anti_momentum_features(market, preds)

    # Train or load model
    if retrain:
        model = train_anti_momentum_model(enhanced)
    else:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Loaded saved model from {MODEL_PATH}")
        else:
            logger.warning("No saved model — retraining")
            model = train_anti_momentum_model(enhanced)

    if model is None:
        return pd.DataFrame()

    # Get latest cross-section (one row per ticker, most recent month)
    latest_month = enhanced["month"].max()
    current = enhanced[enhanced["month"] == latest_month].copy()
    current = current.drop_duplicates("ticker", keep="last")

    # Predict
    features = current[ANTI_MOMENTUM_FEATURES].copy()
    valid_mask = features.notna().all(axis=1)
    current = current[valid_mask].copy()
    features = features[valid_mask]

    if len(current) < 20:
        logger.error(f"Only {len(current)} tickers with valid features — need at least 20")
        return pd.DataFrame()

    current["predicted_return"] = model.predict(features)

    # Rank and generate signals
    current["rank"] = current["predicted_return"].rank(pct=True)
    current["signal"] = (current["rank"] - 0.5) * 2  # [-1, +1]

    # Top N longs, bottom N shorts
    current = current.sort_values("predicted_return", ascending=False)
    n_tickers = len(current)

    current["recommendation"] = "HOLD"
    current.iloc[:top_n, current.columns.get_loc("recommendation")] = "BUY"
    current.iloc[-top_n:, current.columns.get_loc("recommendation")] = "SELL"

    # Confidence: based on how extreme the signal is
    current["confidence"] = current["signal"].abs()

    # Output
    output = current[
        ["ticker", "date", "signal", "predicted_return", "recommendation",
         "confidence", "dist_52w_high", "reversal_5d"]
    ].reset_index(drop=True)

    # Save
    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(SIGNALS_PATH, index=False)
    logger.info(f"Saved {len(output)} signals to {SIGNALS_PATH}")

    # Log summary
    buys = output[output["recommendation"] == "BUY"]
    sells = output[output["recommendation"] == "SELL"]
    logger.info(
        f"Signals: {len(buys)} BUY, {len(sells)} SELL, "
        f"{len(output) - len(buys) - len(sells)} HOLD"
    )

    if len(buys) > 0:
        logger.info(f"Top longs:  {', '.join(buys['ticker'].head(5).tolist())}")
    if len(sells) > 0:
        logger.info(f"Top shorts: {', '.join(sells['ticker'].head(5).tolist())}")

    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Anti-Momentum signal generator")
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh market data and event signals before generating")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of long/short positions (default: 10)")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Use saved model instead of retraining")
    args = parser.parse_args()

    if args.refresh:
        refresh_market_data()
        refresh_event_signals()
        refresh_regime()

    signals = generate_signals(top_n=args.top_n, retrain=not args.no_retrain)

    if not signals.empty:
        print("\n--- Anti-Momentum Signals ---")
        print(f"Date: {signals['date'].iloc[0]}")
        print(f"\nBUY (Long):")
        buys = signals[signals["recommendation"] == "BUY"]
        for _, row in buys.iterrows():
            print(
                f"  {row['ticker']:>6s}  signal={row['signal']:+.3f}  "
                f"pred_ret={row['predicted_return']:+.4f}  "
                f"52w_dist={row['dist_52w_high']:+.2%}  "
                f"rev_5d={row['reversal_5d']:+.2%}"
            )

        print(f"\nSELL (Short):")
        sells = signals[signals["recommendation"] == "SELL"]
        for _, row in sells.iterrows():
            print(
                f"  {row['ticker']:>6s}  signal={row['signal']:+.3f}  "
                f"pred_ret={row['predicted_return']:+.4f}  "
                f"52w_dist={row['dist_52w_high']:+.2%}  "
                f"rev_5d={row['reversal_5d']:+.2%}"
            )


if __name__ == "__main__":
    main()
