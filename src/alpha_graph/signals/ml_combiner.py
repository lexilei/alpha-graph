"""LightGBM walk-forward combiner: predicts 21-day forward returns from the
factor panel (see FACTORS.md). 12-month train window, 1-month test, 31-day
purge (labels span 21 trading days). Also builds the shared feature panel.

Usage:
    python -m alpha_graph.signals.ml_combiner [--train] [--predict]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
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

# Model persistence. LightGBM models are stored in LightGBM's native text
# format — no pickle on the load path. The sklearn GradientBoosting fallback
# (lightgbm missing) has no native format and still pickles, loudly, to a
# separate filename. The pre-2026-07 pickle (ml_combiner_model.pkl) is
# deliberately NOT migrated: it is ignored with a warning; retrain to
# produce the native file.
MODEL_FILENAME_NATIVE = "ml_combiner_model.txt"
MODEL_FILENAME_SKLEARN = "ml_combiner_model_sklearn.pkl"
MODEL_FILENAME_LEGACY = "ml_combiner_model.pkl"


def _save_model(model) -> Path:
    """Persist a trained combiner model under CACHE_DIR.

    LGBMRegressor (the normal case) is saved via its underlying Booster in
    LightGBM's native text format; a raw Booster is saved the same way.
    The sklearn fallback model keeps pickle behind an explicit warning.
    """
    booster = getattr(model, "booster_", None)  # sklearn wrapper (LGBMRegressor)
    if booster is None:
        lgb = _get_lgb()
        if lgb is not None and isinstance(model, lgb.Booster):
            booster = model

    if booster is not None:
        path = CACHE_DIR / MODEL_FILENAME_NATIVE
        path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(path))
        return path

    # No lightgbm: sklearn GradientBoosting fallback has no native format.
    import pickle
    logger.warning("pickle fallback, sklearn model: saving via pickle "
                   f"({MODEL_FILENAME_SKLEARN})")
    path = CACHE_DIR / MODEL_FILENAME_SKLEARN
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def _load_model():
    """Load the saved combiner model, or None if no usable file exists.

    Prefers the native LightGBM text file (loaded as lgb.Booster — for
    regression its predict() matches LGBMRegressor.predict exactly; no
    sklearn-side transform applies). Falls back to the sklearn pickle
    only if that is what was saved. Legacy pickles are ignored.
    """
    native_path = CACHE_DIR / MODEL_FILENAME_NATIVE
    sklearn_path = CACHE_DIR / MODEL_FILENAME_SKLEARN
    legacy_path = CACHE_DIR / MODEL_FILENAME_LEGACY

    if native_path.exists():
        lgb = _get_lgb()
        if lgb is None:
            logger.error(f"lightgbm required to load {native_path}")
            return None
        return lgb.Booster(model_file=str(native_path))

    if sklearn_path.exists():
        import pickle
        logger.warning("pickle fallback, sklearn model: loading from "
                       f"{sklearn_path}")
        with open(sklearn_path, "rb") as f:
            return pickle.load(f)

    if legacy_path.exists():
        logger.warning(f"stale pickle model ignored; retrain: {legacy_path}")
    return None


# --------------------------------------------------------------------------- #
# Factor-source registry: the declarative merge layer of build_feature_panel
# --------------------------------------------------------------------------- #

# Merge kinds (availability semantics; the merge loop is their single owner):
#   "filing"      — filing-date as-of merge; availability_lag_days applies on
#                   EVERY grid (a filing dated t is usable from t + lag).
#   "grid"        — grid-stamped series (builders stamp the computation date);
#                   the lag applies ONLY under daily_signals — on the monthly
#                   grid a +1d shift on a month-end-stamped series pushes it
#                   to the NEXT month-end (the already-tested "stale" variant,
#                   not t+1), so the monthly grid merges unlagged.
#   "grid_static" — monthly-only rejected/never-lagged factors, merged
#                   unlagged on every grid per their documented reasons
#                   (C8/C9 spillover, monthly C11).
VALID_KINDS = ("filing", "grid", "grid_static")


def _load_cos_latest_filing(path: Path) -> pd.DataFrame | None:
    """C7 cos_latest_filing (factor 15): most recent filing's YoY cosine
    (10-K ∪ 10-Q, CMN 2020) — the union of C1's 10-K pairs and C2's 10-Q
    pairs, raw scores pooled; a same-day 10-K + 10-Q keeps the last row.
    Needs BOTH pair caches (`path` is the 10-K cache; the 10-Q cache sits
    next to it)."""
    q_path = path.parent / "lazy_prices_10Q_yoy.parquet"
    if not (path.exists() and q_path.exists()):
        return None
    k = pd.read_parquet(path)[["ticker", "filing_date", "cosine_similarity"]]
    q = pd.read_parquet(q_path)[["ticker", "filing_date", "cos_10q_yoy"]]
    comb = pd.concat([
        k.rename(columns={"cosine_similarity": "cos_latest_filing"}),
        q.rename(columns={"cos_10q_yoy": "cos_latest_filing"}),
    ], ignore_index=True)
    comb["filing_date"] = pd.to_datetime(comb["filing_date"])
    comb = comb.sort_values(["ticker", "filing_date"]).drop_duplicates(
        ["ticker", "filing_date"], keep="last"
    )
    return comb


def _load_sue_pead(path: Path) -> pd.DataFrame | None:
    """C17 sue_pead: as-filed diluted XBRL EPS SUE (disclosure_date from the
    builder: last Item-2.02 8-K date else statement filed date, acceptance-
    time refined; see FACTORS.md C17). Same-day multi-quarter disclosures
    keep the latest period_end; NaN-SUE rows are dropped so they never blank
    an earlier value's carry-forward."""
    if not path.exists():
        return None
    su = pd.read_parquet(path)
    su["disclosure_date"] = pd.to_datetime(su["disclosure_date"])
    su = su.rename(columns={"sue": "sue_pead"}).dropna(subset=["sue_pead"])
    # same-day multi-quarter disclosures: keep the latest period_end
    su = su.sort_values(["ticker", "disclosure_date", "period_end"])
    su = su.drop_duplicates(["ticker", "disclosure_date"], keep="last")
    return su[["ticker", "disclosure_date", "sue_pead"]]


@dataclass(frozen=True)
class FactorSource:
    """One merged signal of build_feature_panel — a declarative registry row.

    Adding a factor to the panel = appending ONE row to FACTOR_SOURCES (and
    registering the ID in FACTORS.md). Fields:

    label:          registry ID(s) for logs (FACTORS.md key, e.g. "C17").
    filename:       cache file under CACHE_DIR.
    cols:           column(s) the merge adds to the panel.
    date_col:       date column in the cache (as-of key at the merge).
    kind:           one of VALID_KINDS — availability-lag semantics (above).
    daily_filename: under daily_signals load THIS cache instead (C10's daily
                    port: same registry ID, same construction, finer grid).
    daily_only:     merge only under daily_signals; no monthly counterpart
                    exists, so the column is ABSENT on the monthly grid (C15).
    dropna:         drop cache rows NaN in cols before the as-of merge ("any"
                    or "all") so a NaN row never blanks a carried-forward
                    value; None keeps all rows.
    loader:         optional preprocessing hook: takes the resolved cache
                    path, returns the ready-to-merge frame (columns: ticker,
                    date_col, *cols) or None for "source unavailable".
                    Covers non-default constructions (C7's union of two
                    caches, C17's same-day dedup). When None, the default
                    load applies: read parquet -> parse date_col -> select
                    [ticker, date_col, *cols] -> dropna -> sort.
    missing_msg:    custom debug message when the source is unavailable.
    """

    label: str
    filename: str
    cols: tuple[str, ...]
    date_col: str
    kind: str
    daily_filename: str | None = None
    daily_only: bool = False
    dropna: str | None = None
    loader: Callable[[Path], pd.DataFrame | None] | None = None
    missing_msg: str | None = None

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"{self.label}: invalid kind {self.kind!r}")
        if self.dropna not in (None, "any", "all"):
            raise ValueError(f"{self.label}: invalid dropna {self.dropna!r}")


# The panel's merged-signal registry, in merge order. To add a factor: build
# its cache, append one FactorSource row here, and register the ID in
# FACTORS.md; evaluation stays the judge's job (scripts/factor_orthogonality.py).
FACTOR_SOURCES: list[FactorSource] = [
    # C1: Lazy Prices TF-IDF cosine between consecutive same-type 10-Ks.
    FactorSource("C1", "lazy_prices_signal_10K.parquet",
                 ("cosine_similarity",), "filing_date", "filing"),
    # C2-C6: the remaining text-factor caches (factors 10-14).
    FactorSource("C2", "lazy_prices_10Q_yoy.parquet",
                 ("cos_10q_yoy",), "filing_date", "filing"),
    FactorSource("C3", "embed_sim_10k_fin.parquet",
                 ("embed_sim_10k_fin",), "filing_date", "filing"),
    FactorSource("C4", "tone_10k.parquet",
                 ("tone_shift_10k",), "filing_date", "filing"),
    FactorSource("C5", "embed_sim_10k_bge.parquet",
                 ("embed_sim_10k_bge",), "filing_date", "filing"),
    FactorSource("C6", "change_detect_10k.parquet",
                 ("new_content_frac",), "filing_date", "filing"),
    # C7 (factor 15): most recent filing's YoY cosine, 10-K ∪ 10-Q union.
    FactorSource("C7", "lazy_prices_10K.parquet",
                 ("cos_latest_filing",), "filing_date", "filing",
                 loader=_load_cos_latest_filing,
                 missing_msg="Factor 15 needs both 10-K and 10-Q pair caches — NaN"),
    # C8/C9 (factors 18-19): graph spillover (computed on the feat/graph-signal
    # branch, shared via data/cache; month-end grid, as-of carried forward;
    # NaN = no scored graph neighbors at that date). Rejected factors:
    # monthly cache only, no daily port, no lag (see the monthly-grid caveat
    # in the build_feature_panel docstring).
    FactorSource("C8/C9", "graph_spillover.parquet",
                 ("spillover_event", "spillover_momentum"), "date",
                 "grid_static", dropna="all"),
    # C10: customers-only momentum spillover (C-F asymmetric). daily_signals
    # selects the daily as-of cache (same construction, finer grid — same
    # registry ID); only the daily grid can express the t+1 lag.
    FactorSource("C10", "graph_customer_momentum.parquet",
                 ("spillover_cust_mom",), "date", "grid",
                 daily_filename="graph_customer_momentum_daily.parquet",
                 dropna="any"),
    # C11: abnormal 8-K filing frequency (month-end z-score, as-of). Always
    # merged unlagged: the headline fresh-month result attaches the month-m z
    # at month-end m; a +1d shift would turn it into the (already tested,
    # near-zero) stale variant. Intramonth availability is C15's job.
    FactorSource("C11", "event_freq_8k.parquet",
                 ("evt8k_freq_z",), "date", "grid_static"),
    # C15: daily 8-K abnormal frequency (NEW variant, not a port of C11:
    # daily grid + trailing-21-trading-day window + lagged non-overlapping
    # baseline; see FACTORS.md). Daily-only: absent on the monthly grid.
    FactorSource("C15", "event_freq_8k_daily.parquet",
                 ("evt8k_freq_z_d",), "date", "grid", daily_only=True),
    # C17: SUE / PEAD. Filing-date-style as-of merge: the availability lag
    # applies on every grid, like the other filing merges (v0: t+1).
    FactorSource("C17", "sue_pead.parquet",
                 ("sue_pead",), "disclosure_date", "filing",
                 loader=_load_sue_pead),
    # C21/C22: paper-faithful Lazy Prices edit measures on consecutive 10-Ks
    # (difflib Ratcliff-Obershelp ratio + CMN Sim_Simple, one pinned token
    # stream; signals/lazy_prices_edit.py). Same 10-K pair set and filing-date
    # availability as C1 — plain filing-date as-of merge, lagged on every grid
    # (v0: t+1). Both columns merge from the one cache.
    FactorSource("C21/C22", "lazy_prices_edit.parquet",
                 ("sim_minedit_10k", "sim_simple_10k"), "filing_date", "filing"),
]


def build_feature_panel(
    pit_universe: bool = True,
    availability_lag_days: int = 1,
    daily_signals: bool = True,
    lag_controls: bool = False,
) -> pd.DataFrame:
    """Assemble all available signals into a single feature panel.

    Merges data from multiple sources using ticker and date as keys.
    Missing features are kept as NaN — LightGBM handles them natively.

    availability_lag_days (Item 2): calendar-day lag between a signal's
    computation date and its first use. Applies to the filing-date merges on
    every grid, and to grid-signal merges ONLY under daily_signals — on the
    monthly grid a +1d shift on a month-end-stamped series pushes it to the
    NEXT month-end, which is the already-tested "stale" variant, not t+1
    (the monthly grid cannot express a one-day lag; that is why Items 2 and
    4 ship jointly).

    daily_signals (Item 4): load the daily as-of caches — C10 customer
    momentum from graph_customer_momentum_daily.parquet (same construction,
    finer grid) and the C15 daily 8-K frequency variant (evt8k_freq_z_d,
    a NEW registry entry; the monthly C11 column stays loaded regardless).

    lag_controls (factorial sensitivity): shift the 6 in-panel price/volume
    control columns by ONE trading day within ticker. The controls are
    computed from same-close prices and are otherwise untouched by
    availability_lag_days (which only lags merged signals); this switch
    checks whether candidate-vs-control comparisons move when the controls
    are held to the same t+1 availability standard. Default False.

    v0 convention (frozen 2026-07-13 after the factorial; see
    reports/factorial_v0.md): pit_universe=True, availability_lag_days=1,
    daily_signals=True — the defaults above. lag_controls stays False
    (factorial sensitivity switch, not part of v0; its scope is the 6
    price/volume controls and does not extend to B7-B9).

    B7-B9 PIT fundamentals controls (FACTORS.md): log_mktcap_pit is a
    same-close price feature (PIT shares, filed <= t, x close);
    book_to_market_pit / earnings_yield_pit are filing-date as-of merges of
    fundamentals_pit.parquet (avail_date = the filed date completing each
    value) under availability_lag_days, divided by the market cap at t.

    Adding a factor: build its cache, append ONE FactorSource row to
    FACTOR_SOURCES (cache filename, column(s), cache date column, merge
    kind — see the dataclass docstring), and register the ID in FACTORS.md.
    The merge loop owns availability semantics (builders stamp computation
    dates, never pre-lag); evaluation stays the judge's job
    (scripts/factor_orthogonality.py) — the panel merely carries the column.
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
    # Liquidity control; true PIT market cap is B7 below (B6 stays: liquidity
    # and size are different controls); its job is to catch text factors that
    # secretly rank by company size.
    dv = (panel["close"] * panel["volume"]).replace(0, np.nan)
    panel["log_dollar_volume"] = np.log(
        dv.groupby(panel["ticker"]).transform(lambda s: s.rolling(63).median())
    )

    # B7 log_mktcap_pit: PIT market cap — the latest share count FILED on or
    # before t (shares_asof semantics: filed <= t, no extra lag — a same-close
    # price feature like the other controls; states_table is the shares_asof
    # collapse for all tickers in one global as-of merge) times the same-day
    # close, logged. BRK-B is excluded (Class-A-equivalent weighted-average
    # series — wrong units for the B-share price; see data/shares_pit.py);
    # GOOG/GOOGL both carry the company-total series (duplication is fine for
    # a control). Closes are split-adjusted, so caps before a ticker's LATER
    # splits are understated by the split factor (market_cap_panel's caveat) —
    # acceptable for a size control. The unlogged cap is kept (_mktcap_pit) as
    # the B8/B9 denominator and dropped before return.
    shares_path = CACHE_DIR / "shares_outstanding_pit.parquet"
    if shares_path.exists():
        from alpha_graph.data.shares_pit import (
            CAP_UNIT_MISMATCH_TICKERS,
            states_table,
        )
        states = states_table(pd.read_parquet(shares_path),
                              exclude_tickers=CAP_UNIT_MISMATCH_TICKERS)
        panel = _merge_asof_signal(
            panel, states.rename(columns={"shares": "_shares_pit"}),
            left_date="date", right_date="filed",
            on="ticker", cols=["_shares_pit"],
        )
        cap = panel["_shares_pit"] * panel["close"]
        panel["_mktcap_pit"] = cap.where(cap > 0)
        panel["log_mktcap_pit"] = np.log(panel["_mktcap_pit"])
        panel = panel.drop(columns=["_shares_pit"])
        # restore the (ticker, date) order the lag_controls/PIT steps assume
        panel = panel.sort_values(["ticker", "date"])
    else:
        panel["_mktcap_pit"] = np.nan
        panel["log_mktcap_pit"] = np.nan
        logger.debug("No shares_outstanding_pit.parquet — B7 will be NaN")

    panel = panel.drop(columns=["daily_ret", "close", "volume"], errors="ignore")

    # Optional t+1 availability for the in-panel controls (factorial
    # sensitivity axis). Must run AFTER all price features are computed but
    # BEFORE the PIT mask below — shifting after row removal would carry
    # values across membership gaps (same ordering reason as the review-F1
    # fix on the mask itself). Panel is still sorted by (ticker, date) here,
    # so shift(1) within ticker is exactly one trading day.
    if lag_controls:
        control_cols = [
            "momentum_21d", "momentum_5d", "volatility_21d",
            "volume_zscore", "momentum_252_21", "log_dollar_volume",
        ]
        panel[control_cols] = panel.groupby("ticker")[control_cols].shift(1)

    # --- PIT universe mask, applied AFTER price-feature computation so
    # lookbacks never span membership gaps (review F1: masking first gave
    # re-entrant names +297% "momentum" across their gap and NaN'd every
    # joiner's first year of 12-1 momentum). Forward-bias fix only; survivor
    # bias remains — see data/pit_universe.py. Default True (v0, 2026-07-13).
    if pit_universe:
        from alpha_graph.data.pit_universe import load_membership, membership_mask
        mask = membership_mask(panel, load_membership())
        n0 = len(panel)
        panel = panel[mask.values].copy()
        logger.info(f"PIT universe: {n0} -> {len(panel)} rows "
                    f"({n0 - len(panel)} non-member rows removed)")

    # Sector labels (current GICS snapshot — mildly non-PIT; used as controls,
    # not as a ranking factor)
    sector_path = CACHE_DIR / "sector_map.parquet"
    if sector_path.exists():
        sectors = pd.read_parquet(sector_path)
        panel = panel.merge(sectors[["ticker", "sector"]], on="ticker", how="left")
        panel["sector"] = panel["sector"].fillna("UNKNOWN")
    else:
        panel["sector"] = "UNKNOWN"

    # --- Merge every registered factor source (FACTOR_SOURCES, in order) ---
    # Missing cache -> NaN column(s) + a debug log. kind picks the lag:
    # "filing" lags on every grid, "grid" lags only under daily_signals,
    # "grid_static" never lags — rationales live on the registry rows.
    for src in FACTOR_SOURCES:
        if src.daily_only and not daily_signals:
            continue  # e.g. C15: no monthly counterpart, column stays absent
        fname = (src.daily_filename if daily_signals and src.daily_filename
                 else src.filename)
        path = CACHE_DIR / fname
        if src.loader is not None:
            sig = src.loader(path)
        elif path.exists():
            sig = pd.read_parquet(path)
            sig[src.date_col] = pd.to_datetime(sig[src.date_col])
            sig = sig[["ticker", src.date_col] + list(src.cols)]
            if src.dropna:
                sig = sig.dropna(subset=list(src.cols), how=src.dropna)
            sig = sig.sort_values(["ticker", src.date_col])
        else:
            sig = None

        if sig is None:
            for c in src.cols:
                panel[c] = np.nan
            logger.debug(src.missing_msg
                         or f"No {fname} — {src.label} {list(src.cols)} will be NaN")
            continue

        if src.kind == "filing":
            lag = availability_lag_days
        elif src.kind == "grid":
            lag = availability_lag_days if daily_signals else 0
        else:  # "grid_static"
            lag = 0
        panel = _merge_asof_signal(
            panel, sig,
            left_date="date", right_date=src.date_col,
            on="ticker", cols=list(src.cols),
            availability_lag_days=lag,
        )

    # --- B8/B9 PIT fundamentals controls: as-filed book equity and trailing
    # 4-quarter net income (fundamentals_pit.parquet), each attached as-of its
    # avail_date = the FILED date of the filing that completed the value (TTM:
    # max of the 4 constituents' filed dates), availability-lagged like every
    # other filing merge (v0: t+1), then divided by the PIT market cap at t
    # (B7's unlogged cap). Negative book equity is real and kept. NaN wherever
    # the numerator has no filed value yet or the cap is unavailable (e.g.
    # BRK-B, or no share count filed yet). ---
    fund_path = CACHE_DIR / "fundamentals_pit.parquet"
    if fund_path.exists():
        fu = pd.read_parquet(fund_path)
        fu["avail_date"] = pd.to_datetime(fu["avail_date"])
        for col in ["book_equity", "net_income_ttm"]:
            ev = fu.dropna(subset=[col])[["ticker", "avail_date", col]]
            ev = ev.sort_values(["ticker", "avail_date"])
            panel = _merge_asof_signal(
                panel, ev,
                left_date="date", right_date="avail_date",
                on="ticker", cols=[col],
                availability_lag_days=availability_lag_days,
            )
        panel["book_to_market_pit"] = panel["book_equity"] / panel["_mktcap_pit"]
        panel["earnings_yield_pit"] = panel["net_income_ttm"] / panel["_mktcap_pit"]
        panel = panel.drop(columns=["book_equity", "net_income_ttm"])
    else:
        panel["book_to_market_pit"] = np.nan
        panel["earnings_yield_pit"] = np.nan
        logger.debug("No fundamentals_pit.parquet — B8/B9 will be NaN")
    panel = panel.drop(columns=["_mktcap_pit"])

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
    availability_lag_days: int = 0,
) -> pd.DataFrame:
    """Merge signal data as-of join: for each panel row, carry forward latest signal.

    This is critical for avoiding lookahead bias — we only use signals
    that were available at the time of the observation.

    availability_lag_days (plan Item 2): a signal dated t is treated as
    available from t + lag CALENDAR days, so with lag=1 it first attaches at
    the next trading close (Friday -> Monday). The merge layer is the SINGLE
    owner of this lag; builders always stamp rows with the computation date.
    """
    panel = panel.sort_values(left_date)
    signal = signal.sort_values(right_date)

    # Normalize datetime dtypes (yfinance returns ms, filings use ns)
    signal = signal.rename(columns={right_date: left_date})
    signal[left_date] = pd.to_datetime(signal[left_date]).astype("datetime64[ns]")
    if availability_lag_days:
        signal[left_date] = signal[left_date] + pd.Timedelta(days=availability_lag_days)
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

        # Save the latest model for research diagnostics (predict_current)
        model_path = _save_model(models[-1]["model"])
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


STALENESS_MAX_DAYS = 7  # calendar days before predict_current warns


def _staleness_days(panel_max_date: pd.Timestamp, now: pd.Timestamp) -> int:
    """Calendar days between the panel's max date and `now`, floored at 0.

    Pure helper so the predict_current staleness guard is testable
    without building the full feature panel.
    """
    gap = pd.Timestamp(now) - pd.Timestamp(panel_max_date)
    return max(0, gap.days)


def predict_current(
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Score the latest panel rows with the saved model — RESEARCH
    DIAGNOSTICS ONLY. NEVER wire this to a broker.

    This output is not a live trading signal, for three concrete reasons:

    1. Stale by construction: build_feature_panel() drops rows via
       ``dropna(subset=["fwd_return_21d"])`` before this function takes
       the per-ticker "latest" row, so that row is the last date whose
       21-trading-day forward return is already known — about 21 trading
       days behind the last cached close, never today.
    2. Frozen, inconsistent caches: the market cache ends 2026-04-02
       while some signal caches carry later dates; nothing here
       refreshes data, so the cross-section is whatever the caches last
       held, merged as-of across mismatched end dates.
    3. Feature drift: FEATURE_COLS is the 5-feature legacy set and
       excludes every SEC candidate factor (cos_10q_yoy, embed_sim_*,
       tone_shift_10k, new_content_frac, cos_latest_filing, spillover_*,
       evt8k_*), so the model scores a subset of the panel actually
       studied in FACTORS.md.

    Guards: if the panel's max date is more than STALENESS_MAX_DAYS
    calendar days behind now, a warning is logged (nothing is raised —
    research repo). The returned frame carries an ``asof_date`` column
    (the panel max date used) so no caller can mistake it for today's
    signal.
    """
    model = _load_model()
    if model is None:
        logger.error("No saved model. Run training first: python -m alpha_graph.signals.ml_combiner --train")
        return pd.DataFrame()

    # Build current feature panel
    panel = build_feature_panel()
    if panel.empty:
        return pd.DataFrame()

    # Staleness guard: the panel ends where the target-labelled data ends,
    # not today. Warn when the gap exceeds the threshold.
    asof_date = panel["date"].max()
    stale_days = _staleness_days(asof_date, pd.Timestamp.now())
    if stale_days > STALENESS_MAX_DAYS:
        logger.warning(
            f"predict_current panel max date {asof_date.date()} is "
            f"{stale_days} calendar days behind now: "
            f"STALE — research only, not a live signal"
        )

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
    # Stamp the panel max date so no caller mistakes this for today's signal
    result["asof_date"] = asof_date
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
