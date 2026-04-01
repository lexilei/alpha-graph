"""Risk management extensions for the ML combiner portfolio.

Four independent overlays that wrap the baseline signal without modifying it:
1. Anti-momentum features → retrained combiner
2. Sector-neutral portfolio construction
3. Regime-conditional models
4. Momentum factor hedge

Usage:
    python -m alpha_graph.backtest.extensions
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_data():
    """Load all cached data needed for extensions."""
    preds = pd.read_parquet(CACHE_DIR / "ml_combiner_predictions.parquet")
    market = pd.read_parquet(CACHE_DIR / "market_data.parquet")
    regimes = pd.read_parquet(CACHE_DIR / "regimes.parquet")
    preds["month"] = preds["date"].dt.to_period("M")
    preds = preds.dropna(subset=["actual_return", "signal"])
    return preds, market, regimes


def _run_ls(preds: pd.DataFrame, long_exp=None, short_exp=None,
            top_n=10, sector_neutral=False, sector_map=None):
    """Run long/short backtest with optional exposure scaling and sector neutrality."""
    months = sorted(preds["month"].unique())
    results = []
    for month in months:
        m = preds[preds["month"] == month].drop_duplicates("ticker", keep="last")
        if len(m) < 20:
            continue

        if sector_neutral and sector_map is not None:
            # Rank within each sector, then pick top/bottom across sectors
            m = m.copy()
            m["sector"] = m["ticker"].map(sector_map).fillna("Unknown")
            m["sector_rank"] = m.groupby("sector")["signal"].rank(pct=True)
            m_sorted = m.sort_values("sector_rank", ascending=False)
        else:
            m_sorted = m.sort_values("signal", ascending=False)

        longs = m_sorted.head(top_n)
        shorts = m_sorted.tail(top_n)

        long_r = longs["actual_return"].mean()
        short_r = shorts["actual_return"].mean()

        ms = str(month)
        le = long_exp.get(ms, 1.0) if long_exp else 1.0
        se = short_exp.get(ms, 1.0) if short_exp else 1.0

        net = le * long_r - se * short_r - 0.002
        results.append({
            "month": month.to_timestamp(), "net": net,
            "long": long_r, "short": short_r, "le": le, "se": se,
        })

    df = pd.DataFrame(results)
    return df


def _metrics(df: pd.DataFrame) -> dict:
    """Compute standard performance metrics from monthly returns."""
    cum = (1 + df["net"]).prod() - 1
    sharpe = df["net"].mean() / df["net"].std() * np.sqrt(12) if df["net"].std() > 0 else 0
    maxdd = ((1 + df["net"]).cumprod() / (1 + df["net"]).cumprod().cummax() - 1).min()
    win = (df["net"] > 0).mean()
    ann = (1 + cum) ** (12 / len(df)) - 1
    return {"cum": cum, "sharpe": sharpe, "maxdd": maxdd, "win": win, "ann": ann}


# ---------------------------------------------------------------------------
# Extension 1: Anti-Momentum Features
# ---------------------------------------------------------------------------

def ext1_anti_momentum_features(preds, market, regimes):
    """Retrain ML combiner with anti-momentum features added.

    New features:
    - dist_52w_high: distance from 52-week high (mean reversion signal)
    - reversal_5d: negative of 5-day return (short-term reversal)
    - vol_regime_interaction: volatility * regime (conditional risk)
    """
    logger.info("Extension 1: Anti-Momentum Features")

    market = market.sort_values(["ticker", "date"]).copy()

    # 52-week high distance: (price / 252-day max) - 1
    market["high_252d"] = market.groupby("ticker")["close"].transform(
        lambda s: s.rolling(252, min_periods=63).max()
    )
    market["dist_52w_high"] = market["close"] / market["high_252d"] - 1

    # Short-term reversal (negative momentum)
    market["reversal_5d"] = -market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(5)
    )

    # Merge new features into predictions
    market["month"] = market["date"].dt.to_period("M")
    new_feat = market.groupby(["ticker", "month"]).last()[
        ["dist_52w_high", "reversal_5d"]
    ].reset_index()

    enhanced = preds.merge(new_feat, on=["ticker", "month"], how="left")

    # Retrain a simple LightGBM with expanded features
    try:
        import lightgbm as lgb
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor as lgb
        lgb = None

    feature_cols = ["signal", "dist_52w_high", "reversal_5d"]
    train_results = []
    months = sorted(enhanced["month"].unique())

    for i, test_month in enumerate(months):
        train_months = months[max(0, i - 12):i]
        if len(train_months) < 6:
            continue

        train = enhanced[enhanced["month"].isin(train_months)].dropna(
            subset=feature_cols + ["actual_return"]
        )
        test = enhanced[enhanced["month"] == test_month].dropna(
            subset=feature_cols + ["actual_return"]
        )
        if len(train) < 100 or len(test) < 10:
            continue

        X_tr, y_tr = train[feature_cols].values, train["actual_return"].values
        X_te = test[feature_cols].values

        if lgb is not None:
            model = lgb.LGBMRegressor(
                n_estimators=150, num_leaves=15, max_depth=5,
                learning_rate=0.03, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
                verbose=-1,
            )
        else:
            model = lgb(n_estimators=150, max_depth=5, learning_rate=0.03)

        model.fit(X_tr, y_tr)
        test = test.copy()
        test["enhanced_signal"] = model.predict(X_te)
        train_results.append(test)

    if not train_results:
        logger.warning("Extension 1: not enough data to retrain")
        return _run_ls(preds), "Anti-Momentum Features"

    enhanced_preds = pd.concat(train_results, ignore_index=True)
    enhanced_preds["signal"] = enhanced_preds["enhanced_signal"]
    df = _run_ls(enhanced_preds)
    return df, "Anti-Momentum Features"


# ---------------------------------------------------------------------------
# Extension 2: Sector-Neutral Portfolio
# ---------------------------------------------------------------------------

def ext2_sector_neutral(preds, market, regimes):
    """Construct sector-neutral long/short portfolio.

    Instead of ranking globally, rank within each sector and pick
    top/bottom from each sector proportionally.
    """
    logger.info("Extension 2: Sector-Neutral Portfolio")

    # Build sector map from 8-K filing directory structure
    # Fallback: use simple GICS-like mapping
    sector_map = _build_sector_map(preds["ticker"].unique())

    df = _run_ls(preds, sector_neutral=True, sector_map=sector_map)
    return df, "Sector-Neutral"


def _build_sector_map(tickers):
    """Simple sector mapping for S&P 500 tickers."""
    # Hardcoded major sectors for the tickers we know
    tech = {"AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "AVGO", "ADBE", "AMD", "AMAT",
            "ADSK", "ANET", "CDNS", "AKAM", "ADI", "APP", "AXON", "TECH"}
    finance = {"BRK-B", "BLK", "BX", "BAC", "COF", "AXP", "AIG", "ALL", "AFL",
               "ACGL", "AIZ", "AMP", "APO", "ARES", "BK", "BRO", "CBOE", "SCHW"}
    health = {"ABBV", "ABT", "AMGN", "BIIB", "BAX", "BDX", "BMY", "BSX", "CNC",
              "COR", "CRL", "ALGN"}
    consumer = {"AMZN", "ABNB", "BKNG", "CCL", "BBY", "CPB", "CVNA", "CDW"}
    industrial = {"BA", "CAT", "APH", "APTV", "CARR", "AME", "AOS", "ALLE", "BLDR",
                  "BG", "CHRW", "A"}
    energy = {"APA", "AES", "AEP", "AEE", "BKR", "CF", "ATO", "AWK", "CNP", "LNT"}
    realestate = {"AMT", "ARE", "AVB", "BXP", "CPT"}
    comm = {"T", "CHTR", "GOOGL", "GOOG"}
    materials = {"ALB", "APD", "AMCR", "AVY", "BALL", "ADM"}
    other = {"MO", "MMM", "ACN", "ADP", "AJG", "AON", "CAH", "BF-B", "BR", "XYZ"}

    mapping = {}
    for sector_name, sector_set in [
        ("Tech", tech), ("Finance", finance), ("Health", health),
        ("Consumer", consumer), ("Industrial", industrial), ("Energy", energy),
        ("RealEstate", realestate), ("Comm", comm), ("Materials", materials),
        ("Other", other),
    ]:
        for t in sector_set:
            mapping[t] = sector_name

    # Assign Unknown to any unmapped ticker
    for t in tickers:
        if t not in mapping:
            mapping[t] = "Other"

    return mapping


# ---------------------------------------------------------------------------
# Extension 3: Regime-Conditional Models
# ---------------------------------------------------------------------------

def ext3_regime_conditional(preds, market, regimes):
    """Train separate models for each regime.

    Instead of one model for all regimes, train:
    - Trending model: avoids high-beta longs
    - Crisis model: favors defensive names
    - Mean-reverting model: standard signal
    """
    logger.info("Extension 3: Regime-Conditional Models")

    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None

    # Merge regime into predictions
    regimes_monthly = regimes.copy()
    regimes_monthly["month"] = regimes_monthly["date"].dt.to_period("M")
    regime_mode = regimes_monthly.groupby("month")["regime"].agg(
        lambda x: x.mode()[0]
    ).to_dict()

    preds = preds.copy()
    preds["regime"] = preds["month"].map(regime_mode)

    # Add volatility as feature
    market = market.sort_values(["ticker", "date"]).copy()
    market["vol_21d"] = market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change().rolling(21).std() * np.sqrt(252)
    )
    market["month"] = market["date"].dt.to_period("M")
    vol_feat = market.groupby(["ticker", "month"]).last()["vol_21d"].reset_index()
    preds = preds.merge(vol_feat, on=["ticker", "month"], how="left")

    feature_cols = ["signal", "vol_21d"]
    months = sorted(preds["month"].unique())
    all_results = []

    for i, test_month in enumerate(months):
        train_months = months[max(0, i - 12):i]
        if len(train_months) < 6:
            continue

        test = preds[preds["month"] == test_month].dropna(
            subset=feature_cols + ["actual_return"]
        )
        if len(test) < 10:
            continue

        test_regime = test["regime"].mode()[0] if len(test) > 0 else "TRENDING"

        # Train on same-regime data only
        train = preds[
            (preds["month"].isin(train_months)) & (preds["regime"] == test_regime)
        ].dropna(subset=feature_cols + ["actual_return"])

        # Fallback: if too few same-regime samples, use all data
        if len(train) < 50:
            train = preds[preds["month"].isin(train_months)].dropna(
                subset=feature_cols + ["actual_return"]
            )

        if len(train) < 50:
            continue

        X_tr, y_tr = train[feature_cols].values, train["actual_return"].values
        X_te = test[feature_cols].values

        if lgb is not None:
            model = lgb.LGBMRegressor(
                n_estimators=150, num_leaves=15, max_depth=5,
                learning_rate=0.03, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
                verbose=-1,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=150, max_depth=5)

        model.fit(X_tr, y_tr)
        test = test.copy()
        test["signal"] = model.predict(X_te)
        all_results.append(test)

    if not all_results:
        logger.warning("Extension 3: not enough data")
        return _run_ls(preds), "Regime-Conditional"

    regime_preds = pd.concat(all_results, ignore_index=True)
    df = _run_ls(regime_preds)
    return df, "Regime-Conditional"


# ---------------------------------------------------------------------------
# Extension 4: Momentum Factor Hedge
# ---------------------------------------------------------------------------

def ext4_momentum_hedge(preds, market, regimes):
    """Hedge portfolio momentum exposure using a synthetic momentum factor.

    Compute the portfolio's trailing beta to the momentum factor,
    then subtract momentum factor return * beta from net return.
    This simulates hedging with a momentum ETF (e.g., MTUM).
    """
    logger.info("Extension 4: Momentum Factor Hedge")

    market = market.sort_values(["ticker", "date"]).copy()
    market["mom_21d"] = market.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(21)
    )
    market["fwd_21d"] = market.groupby("ticker")["close"].transform(
        lambda s: s.shift(-21) / s - 1
    )
    market["month"] = market["date"].dt.to_period("M")

    # Compute monthly momentum factor return (top10 - bottom10 by momentum)
    mom_factor = {}
    for m in market["month"].unique():
        sub = market[market["month"] == m].drop_duplicates("ticker", keep="last")
        sub = sub.dropna(subset=["mom_21d", "fwd_21d"])
        if len(sub) < 20:
            continue
        s = sub.sort_values("mom_21d", ascending=False)
        mom_ret = s.head(10)["fwd_21d"].mean() - s.tail(10)["fwd_21d"].mean()
        mom_factor[m] = mom_ret

    # Run baseline L/S
    df = _run_ls(preds)

    # Compute rolling beta of portfolio returns to momentum factor
    df["mom_factor"] = df["month"].dt.to_period("M").map(mom_factor)
    df = df.dropna(subset=["mom_factor"])

    hedged_net = []
    for i in range(len(df)):
        if i < 3:
            # Not enough history for beta estimation, no hedge
            hedged_net.append(df.iloc[i]["net"])
            continue

        # Trailing 6-month beta
        lookback = df.iloc[max(0, i - 6):i]
        if len(lookback) < 3 or lookback["mom_factor"].std() == 0:
            hedged_net.append(df.iloc[i]["net"])
            continue

        cov = np.cov(lookback["net"], lookback["mom_factor"])
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
        beta = np.clip(beta, -1.0, 1.0)  # cap hedge ratio

        # Hedged return = portfolio return - beta * momentum factor return
        hedge_cost = 0.001  # 10 bps cost for the hedge trade
        hedged = df.iloc[i]["net"] - beta * df.iloc[i]["mom_factor"] - hedge_cost
        hedged_net.append(hedged)

    df = df.copy()
    df["net"] = hedged_net
    return df, "Momentum Hedge"


# ---------------------------------------------------------------------------
# Runner: compare all extensions against baseline
# ---------------------------------------------------------------------------

def run_all_extensions():
    """Run all 4 extensions and compare against baseline."""
    preds, market, regimes = _load_data()

    # Baseline
    baseline_df = _run_ls(preds)
    baseline_m = _metrics(baseline_df)

    results = [("Baseline", baseline_df, baseline_m)]

    extensions = [
        ext1_anti_momentum_features,
        ext2_sector_neutral,
        ext3_regime_conditional,
        ext4_momentum_hedge,
    ]

    for ext_fn in extensions:
        try:
            df, name = ext_fn(preds.copy(), market.copy(), regimes.copy())
            m = _metrics(df)
            results.append((name, df, m))
        except Exception as e:
            logger.error(f"{ext_fn.__name__}: {e}")

    # Print comparison table
    print("\n" + "=" * 75)
    print("EXTENSION COMPARISON (24-month OOS)")
    print("=" * 75)
    print(f"{'Strategy':<28s} {'Cum':>8s} {'Ann':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Win':>6s}")
    print("-" * 75)
    for name, df, m in results:
        print(
            f"{name:<28s} {m['cum']:>+7.1%} {m['ann']:>+7.1%} "
            f"{m['sharpe']:>+7.2f} {m['maxdd']:>7.1%} {m['win']:>5.0%}"
        )

    # Jan 2025 comparison
    print("\n--- January 2025 ---")
    for name, df, m in results:
        jan = df[df["month"].dt.to_period("M") == "2025-01"]
        if len(jan) > 0:
            print(f"  {name:<28s}: net={jan.iloc[0]['net']:+.2%}")

    return results


if __name__ == "__main__":
    run_all_extensions()
