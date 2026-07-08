"""LightGBM walk-forward combiner: predicts 21-day forward returns from the
factor panel (see FACTORS.md). 12-month train window, 1-month test, 31-day
purge (labels span 21 trading days). Also builds the shared feature panel.

Usage:
    python -m alpha_graph.signals.ml_combiner [--train] [--predict]
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from alpha_graph.config import CACHE_DIR, FILINGS_DIR

# LightGBM imported lazily to allow partial installs
_lgb = None


def _get_lgb():
    """Lazily import lightgbm, falling back to sklearn GradientBoosting."""
    global _lgb
    if _lgb is not None:
        return _lgb

    try:
        import lightgbm as lgb
        _lgb = lgb
        return lgb
    except ImportError:
        logger.warning(
            "lightgbm not installed. Install with: pip install lightgbm. "
            "Falling back to sklearn GradientBoostingRegressor."
        )
        return None


# Model hyperparameters (conservative to prevent overfitting on small universe).
# `deterministic=True` + `n_jobs=1` are required for bit-identical fits across
# runs and machines — without them LightGBM's parallel histogram aggregation
# produces non-reproducible Sharpe ratios.
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 15,          # shallow trees to prevent overfitting
    "learning_rate": 0.05,
    "feature_fraction": 0.7,   # column subsampling
    "bagging_fraction": 0.8,   # row subsampling
    "bagging_freq": 5,
    "min_child_samples": 10,   # minimum data in a leaf
    "reg_alpha": 0.1,          # L1 regularization
    "reg_lambda": 1.0,         # L2 regularization
    "n_estimators": 200,
    "verbose": -1,
    "random_state": 42,
    "deterministic": True,     # bit-reproducible histogram aggregation
    "n_jobs": 1,               # single-threaded for cross-machine determinism
}

# Walk-forward settings
TRAIN_MONTHS = 12
PURGE_DAYS = 31  # labels are 21 TRADING days forward (~31 calendar)
FEATURE_COLS = [
    "cosine_similarity",
    "momentum_21d",
    "momentum_5d",
    "volatility_21d",
    "volume_zscore",
]


def build_feature_panel() -> pd.DataFrame:
    """Assemble all available signals into a single feature panel.

    Merges data from multiple sources using ticker and date as keys.
    Missing features are kept as NaN — LightGBM handles them natively.
    """
    market_path = CACHE_DIR / "market_data.parquet"
    if not market_path.exists():
        logger.error("No market data. Run: python -m alpha_graph.data.market")
        return pd.DataFrame()

    market = pd.read_parquet(market_path)
    market["date"] = pd.to_datetime(market["date"])

    # --- Base: market data with returns and momentum features ---
    panel = market[["ticker", "date", "close", "volume", "ret_21d"]].copy()
    panel = panel.rename(columns={"ret_21d": "fwd_return_21d"})

    # Compute momentum features (backward-looking, no lookahead)
    panel = panel.sort_values(["ticker", "date"])
    panel["momentum_21d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(21)
    )
    panel["momentum_5d"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(5)
    )
    # Factor 16: classic 12-1 momentum — return over t-252..t-21, skipping the
    # most recent month (the standard academic definition; the June audit found
    # the cosine signal resembles exactly this, so it must sit in the controls).
    panel["momentum_252_21"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.shift(21) / s.shift(252) - 1.0
    )

    # Realized volatility (backward-looking)
    panel["daily_ret"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.pct_change()
    )
    panel["volatility_21d"] = panel.groupby("ticker")["daily_ret"].transform(
        lambda s: s.rolling(21).std() * np.sqrt(252)
    )

    # Volume z-score (relative to own history)
    panel["volume_zscore"] = panel.groupby("ticker")["volume"].transform(
        lambda s: (s - s.rolling(63).mean()) / s.rolling(63).std()
    )

    # Factor 17: size/liquidity proxy — log of 63d median dollar volume.
    # NOT true market cap (no PIT shares outstanding yet); its job is to catch
    # text factors that secretly rank by company size.
    dv = (panel["close"] * panel["volume"]).replace(0, np.nan)
    panel["log_dollar_volume"] = np.log(
        dv.groupby(panel["ticker"]).transform(lambda s: s.rolling(63).median())
    )

    panel = panel.drop(columns=["daily_ret", "close", "volume"], errors="ignore")

    # Sector labels (current GICS snapshot — mildly non-PIT; used as controls,
    # not as a ranking factor)
    sector_path = CACHE_DIR / "sector_map.parquet"
    if sector_path.exists():
        sectors = pd.read_parquet(sector_path)
        panel = panel.merge(sectors[["ticker", "sector"]], on="ticker", how="left")
        panel["sector"] = panel["sector"].fillna("UNKNOWN")
    else:
        panel["sector"] = "UNKNOWN"

    # --- Merge Lazy Prices signals ---
    lazy_path = CACHE_DIR / "lazy_prices_signal_10K.parquet"
    if lazy_path.exists():
        lazy = pd.read_parquet(lazy_path)
        lazy["filing_date"] = pd.to_datetime(lazy["filing_date"])
        # For each ticker/date, carry forward the latest cosine_similarity
        lazy = lazy[["ticker", "filing_date", "cosine_similarity"]].copy()
        lazy = lazy.sort_values(["ticker", "filing_date"])
        # Merge as-of: for each row in panel, get most recent lazy signal
        panel = _merge_asof_signal(
            panel, lazy,
            left_date="date", right_date="filing_date",
            on="ticker", cols=["cosine_similarity"],
        )
    else:
        panel["cosine_similarity"] = np.nan
        logger.debug("No Lazy Prices data — feature will be NaN")

    # --- Merge remaining text-factor caches (factors 10-14) ---
    for fname, cols in [
        ("lazy_prices_10Q_yoy.parquet", ["cos_10q_yoy"]),
        ("embed_sim_10k_fin.parquet", ["embed_sim_10k_fin"]),
        ("tone_10k.parquet", ["tone_shift_10k"]),
        ("embed_sim_10k_bge.parquet", ["embed_sim_10k_bge"]),
        ("change_detect_10k.parquet", ["new_content_frac"]),
    ]:
        path = CACHE_DIR / fname
        if not path.exists():
            for c in cols:
                panel[c] = np.nan
            logger.debug(f"No {fname} — {cols} will be NaN")
            continue
        sig = pd.read_parquet(path)
        sig["filing_date"] = pd.to_datetime(sig["filing_date"])
        sig = sig[["ticker", "filing_date"] + cols].sort_values(["ticker", "filing_date"])
        panel = _merge_asof_signal(
            panel, sig,
            left_date="date", right_date="filing_date",
            on="ticker", cols=cols,
        )

    # --- Factor 15: most recent filing's YoY cosine (10-K ∪ 10-Q, CMN 2020) ---
    k_path = CACHE_DIR / "lazy_prices_10K.parquet"
    q_path = CACHE_DIR / "lazy_prices_10Q_yoy.parquet"
    if k_path.exists() and q_path.exists():
        k = pd.read_parquet(k_path)[["ticker", "filing_date", "cosine_similarity"]]
        q = pd.read_parquet(q_path)[["ticker", "filing_date", "cos_10q_yoy"]]
        comb = pd.concat([
            k.rename(columns={"cosine_similarity": "cos_latest_filing"}),
            q.rename(columns={"cos_10q_yoy": "cos_latest_filing"}),
        ], ignore_index=True)
        comb["filing_date"] = pd.to_datetime(comb["filing_date"])
        comb = comb.sort_values(["ticker", "filing_date"]).drop_duplicates(
            ["ticker", "filing_date"], keep="last"
        )
        panel = _merge_asof_signal(
            panel, comb,
            left_date="date", right_date="filing_date",
            on="ticker", cols=["cos_latest_filing"],
        )
    else:
        panel["cos_latest_filing"] = np.nan
        logger.debug("Factor 15 needs both 10-K and 10-Q pair caches — NaN")

    # Drop rows with no target
    panel = panel.dropna(subset=["fwd_return_21d"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    logger.info(
        f"Feature panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers, "
        f"{panel['date'].min().date()} to {panel['date'].max().date()}"
    )

    # Log feature coverage
    for col in FEATURE_COLS:
        if col in panel.columns:
            coverage = panel[col].notna().mean() * 100
            logger.debug(f"  {col}: {coverage:.1f}% non-null")

    return panel


def _merge_asof_signal(
    panel: pd.DataFrame,
    signal: pd.DataFrame,
    left_date: str,
    right_date: str,
    on: str,
    cols: list[str],
) -> pd.DataFrame:
    """Merge signal data as-of join: for each panel row, carry forward latest signal.

    This is critical for avoiding lookahead bias — we only use signals
    that were available at the time of the observation.
    """
    panel = panel.sort_values(left_date)
    signal = signal.sort_values(right_date)

    # Normalize datetime dtypes (yfinance returns ms, filings use ns)
    signal = signal.rename(columns={right_date: left_date})
    signal[left_date] = pd.to_datetime(signal[left_date]).astype("datetime64[ns]")
    panel[left_date] = pd.to_datetime(panel[left_date]).astype("datetime64[ns]")

    merged = pd.merge_asof(
        panel,
        signal,
        on=left_date,
        by=on,
        direction="backward",
    )
    return merged


def walk_forward_train_predict(
    panel: pd.DataFrame,
    train_months: int = TRAIN_MONTHS,
    purge_days: int = PURGE_DAYS,
) -> pd.DataFrame:
    """Walk-forward training and prediction.

    For each month in the panel:
    1. Train on the previous `train_months` of data
    2. Purge `purge_days` between train end and test start
    3. Predict on the current month
    4. Record predictions and actual returns

    Returns DataFrame with: date, ticker, predicted_return, actual_return, signal.
    """
    lgb = _get_lgb()

    panel = panel.copy()
    panel["month"] = panel["date"].dt.to_period("M")
    months = sorted(panel["month"].unique())

    if len(months) < train_months + 2:
        logger.error(
            f"Need at least {train_months + 2} months of data, have {len(months)}"
        )
        return pd.DataFrame()

    # Determine available features (present in data)
    available_features = [c for c in FEATURE_COLS if c in panel.columns]
    logger.info(f"Training with features: {available_features}")

    results = []
    models = []

    for i in range(train_months, len(months)):
        test_month = months[i]
        train_start = months[max(0, i - train_months)]
        train_end = months[i - 1]

        # Training data: train_start to train_end
        train_mask = (panel["month"] >= train_start) & (panel["month"] <= train_end)
        train_data = panel[train_mask].copy()

        # Purge: remove last `purge_days` of training data to prevent leakage
        if purge_days > 0:
            purge_cutoff = train_data["date"].max() - pd.Timedelta(days=purge_days)
            train_data = train_data[train_data["date"] <= purge_cutoff]

        # Test data: test_month only
        test_mask = panel["month"] == test_month
        test_data = panel[test_mask].copy()

        if len(train_data) < 50 or len(test_data) < 5:
            logger.debug(
                f"[{test_month}] Skipping: train={len(train_data)}, test={len(test_data)}"
            )
            continue

        X_train = train_data[available_features]
        y_train = train_data["fwd_return_21d"]
        X_test = test_data[available_features]
        y_test = test_data["fwd_return_21d"]

        # Train model (fixed n_estimators; never early-stop on the test month)
        if lgb is not None:
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(X_train, y_train)
        else:
            # Fallback to sklearn
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            # sklearn doesn't handle NaN — fill with median
            X_train_filled = X_train.fillna(X_train.median())
            X_test_filled = X_test.fillna(X_train.median())
            model.fit(X_train_filled, y_train)
            X_test = X_test_filled

        # Predict
        if lgb is not None:
            predictions = model.predict(X_test)
        else:
            predictions = model.predict(X_test)

        # Record results
        for idx, (_, row) in enumerate(test_data.iterrows()):
            results.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "predicted_return": float(predictions[idx]),
                "actual_return": float(row["fwd_return_21d"]),
            })

        # Log performance for this period
        ic = stats.spearmanr(predictions, y_test.values).statistic
        logger.info(
            f"[{test_month}] Train: {len(train_data)} | Test: {len(test_data)} | "
            f"IC: {ic:.4f}"
        )

        models.append({
            "month": str(test_month),
            "model": model,
            "ic": ic,
            "n_train": len(train_data),
            "n_test": len(test_data),
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.warning("No walk-forward results produced")
        return results_df

    results_df["date"] = pd.to_datetime(results_df["date"])

    # Convert predictions to signal: rank within each date cross-section
    results_df["signal"] = results_df.groupby("date")["predicted_return"].transform(
        lambda s: (s.rank(pct=True) - 0.5) * 2  # maps to [-1, +1]
    )

    # Save model metrics
    if models:
        avg_ic = np.mean([m["ic"] for m in models if not np.isnan(m["ic"])])
        logger.info(f"Walk-forward complete: avg IC = {avg_ic:.4f} over {len(models)} months")

        # Save the latest model for live prediction
        model_path = CACHE_DIR / "ml_combiner_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(models[-1]["model"], f)
        logger.info(f"Saved latest model to {model_path}")

    # Save results
    out_path = CACHE_DIR / "ml_combiner_predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(results_df)} predictions to {out_path}")

    return results_df


def compute_signal_metrics(results: pd.DataFrame) -> dict:
    """Evaluate ML combiner signal quality.

    Key metrics:
    - IC (Information Coefficient): rank correlation between prediction and actual
    - ICIR (IC Information Ratio): IC / std(IC) — consistency of IC
    - Long-short spread: average return of top quintile minus bottom quintile
    """
    if results.empty:
        return {}

    # Overall IC
    overall_ic = stats.spearmanr(
        results["predicted_return"], results["actual_return"]
    ).statistic

    # Monthly IC series
    monthly_ic = results.groupby(results["date"].dt.to_period("M")).apply(
        lambda g: stats.spearmanr(g["predicted_return"], g["actual_return"]).statistic
        if len(g) >= 10 else np.nan
    ).dropna()

    icir = monthly_ic.mean() / monthly_ic.std() if monthly_ic.std() > 0 else 0.0

    # Quintile analysis
    results = results.copy()
    results["quintile"] = results.groupby("date")["predicted_return"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop") + 1
    )
    quintile_returns = results.groupby("quintile")["actual_return"].mean()

    # Long-short spread (Q5 - Q1)
    ls_spread = 0.0
    if 5 in quintile_returns.index and 1 in quintile_returns.index:
        ls_spread = quintile_returns[5] - quintile_returns[1]

    metrics = {
        "overall_ic": float(overall_ic),
        "mean_monthly_ic": float(monthly_ic.mean()),
        "icir": float(icir),
        "n_months": len(monthly_ic),
        "pct_positive_ic": float((monthly_ic > 0).mean()),
        "long_short_spread": float(ls_spread),
    }

    return metrics


def feature_importance(panel: pd.DataFrame) -> pd.DataFrame:
    """Train on full data and extract feature importance for interpretation.

    This is NOT for trading — just for understanding which signals matter most.
    """
    lgb = _get_lgb()
    if lgb is None:
        logger.warning("lightgbm required for feature importance analysis")
        return pd.DataFrame()

    available_features = [c for c in FEATURE_COLS if c in panel.columns]
    X = panel[available_features]
    y = panel["fwd_return_21d"]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance["importance_pct"] = (
        importance["importance"] / importance["importance"].sum() * 100
    )

    return importance


def predict_current(
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Generate current predictions using the latest saved model.

    This is the production inference path — loads the saved model and
    generates predictions for the current cross-section of stocks.
    """
    model_path = CACHE_DIR / "ml_combiner_model.pkl"
    if not model_path.exists():
        logger.error("No saved model. Run training first: python -m alpha_graph.signals.ml_combiner --train")
        return pd.DataFrame()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Build current feature panel
    panel = build_feature_panel()
    if panel.empty:
        return pd.DataFrame()

    # Use only the latest date per ticker
    latest = panel.sort_values("date").groupby("ticker").tail(1).copy()

    if tickers is not None:
        latest = latest[latest["ticker"].isin(tickers)]

    available_features = [c for c in FEATURE_COLS if c in latest.columns]
    X = latest[available_features]

    try:
        predictions = model.predict(X)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return pd.DataFrame()

    latest["ml_predicted_return"] = predictions
    latest["ml_signal"] = (
        latest["ml_predicted_return"].rank(pct=True) - 0.5
    ) * 2  # [-1, +1]

    result = latest[["ticker", "date", "ml_predicted_return", "ml_signal"]].copy()
    result = result.sort_values("ml_signal", ascending=False).reset_index(drop=True)

    # Save
    out_path = CACHE_DIR / "ml_combiner_current.parquet"
    result.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(result)} current ML signals to {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="ML Signal Combiner (LightGBM)")
    parser.add_argument("--train", action="store_true", help="Run walk-forward training")
    parser.add_argument("--predict", action="store_true", help="Generate current predictions")
    parser.add_argument("--importance", action="store_true", help="Show feature importance")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers for prediction")
    args = parser.parse_args()

    if args.train or (not args.predict and not args.importance):
        # Build features and run walk-forward
        panel = build_feature_panel()
        if panel.empty:
            print("No data available for training")
            return

        results = walk_forward_train_predict(panel)
        if results.empty:
            print("Walk-forward produced no results")
            return

        metrics = compute_signal_metrics(results)
        print("\n--- ML Combiner Walk-Forward Results ---")
        print(f"  Overall IC:        {metrics.get('overall_ic', 0):.4f}")
        print(f"  Mean Monthly IC:   {metrics.get('mean_monthly_ic', 0):.4f}")
        print(f"  ICIR:              {metrics.get('icir', 0):.4f}")
        print(f"  % Positive IC:     {metrics.get('pct_positive_ic', 0):.1%}")
        print(f"  L/S Spread (21d):  {metrics.get('long_short_spread', 0):.4f}")
        print(f"  Months tested:     {metrics.get('n_months', 0)}")

        if args.importance:
            imp = feature_importance(panel)
            if not imp.empty:
                print("\n--- Feature Importance ---")
                print(imp.to_string(index=False))

    if args.predict:
        predictions = predict_current(tickers=args.tickers)
        if not predictions.empty:
            print("\n--- Current ML Predictions ---")
            print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
