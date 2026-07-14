"""Gate-2 tradeable backtest of C15 (daily abnormal 8-K frequency), decile L/S.

Runs the design registered pre-run in reports/factor_preregistration.md
(2026-07-13) verbatim — no tuning, no extra variants:

- Signal: v0 feature panel (`build_feature_panel()` pure defaults: PIT
  membership, t+1 availability lag, daily grid), column `evt8k_freq_z_d`,
  non-null rows only. The panel's dropna(fwd_return_21d) trims the final
  ~21 trading days of the sample.
- Portfolio: decile long-short, LONG the lowest-z decile / SHORT the
  highest-z decile (direction=-1; sign per C15's negative IC), equal weight
  within leg, per-name cap 0.05 (non-binding at ~38 names/leg).
- Rebalance: each month's last trading close, using that day's available
  signal (engine as-of join; the panel signal is already availability-lagged).
- Cells (exactly three): base (close execution, half_spread=2bp,
  commission=1bp, borrow=50bp p.a.), doubled costs (same, all components x2),
  next_open execution (base costs, trade at the next day's open).
- Base-cell extras: HAC t of the monthly compounded net return series
  (lags from ic_tools.default_lags(21, 21, ic=series) — the library's
  documented rule for a monthly non-overlapping series), and FF5+MOM daily
  attribution with HAC errors (ff_attribution defaults, lags=32).

Output: JSON to stdout (one object with a key per cell + base extras).
The gate verdicts are read against reports/promotion_gate_c11.md.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR
from alpha_graph.data.pit_universe import membership_mask
from alpha_graph.eval.ff_attribution import ff_attribution
from alpha_graph.eval.ic_tools import default_lags, hac_tstat
from alpha_graph.portfolio import CostModel, run_ls_backtest, summarize
from alpha_graph.signals.ml_combiner import build_feature_panel

SIGNAL_COL = "evt8k_freq_z_d"
N_QUANTILES = 10
DIRECTION = -1  # long the LOWEST-z decile (C15's IC is negative)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if np.isnan(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    panel = build_feature_panel()  # pure defaults = frozen v0 conventions
    sig = (
        panel.loc[panel[SIGNAL_COL].notna(), ["date", "ticker", SIGNAL_COL]]
        .rename(columns={SIGNAL_COL: "value"})
        .copy()
    )
    prices = pd.read_parquet(CACHE_DIR / "market_data.parquet")[
        ["ticker", "date", "close", "open"]
    ]
    eligibility = prices[["ticker", "date"]].copy()
    eligibility["eligible"] = membership_mask(eligibility)
    sector_map = (
        pd.read_parquet(CACHE_DIR / "sector_map.parquet")
        .set_index("ticker")["sector"]
        .to_dict()
    )

    out: dict = {
        "signal": {
            "column": SIGNAL_COL,
            "n_rows": int(len(sig)),
            "n_tickers": int(sig["ticker"].nunique()),
            "first_date": str(sig["date"].min().date()),
            "last_date": str(sig["date"].max().date()),
            "panel_note": "v0 panel dropna(fwd_return_21d) trims the final ~21 trading days",
        },
        "prices": {
            "first_date": str(prices["date"].min().date()),
            "last_date": str(prices["date"].max().date()),
            "n_tickers": int(prices["ticker"].nunique()),
        },
    }

    cells = {
        "base": dict(execution="close", costs=CostModel(2.0, 1.0, 50.0)),
        "doubled_costs": dict(execution="close",
                              costs=CostModel(2.0, 1.0, 50.0, doubled=True)),
        "next_open": dict(execution="next_open", costs=CostModel(2.0, 1.0, 50.0)),
    }

    base_result = None
    for name, spec in cells.items():
        res = run_ls_backtest(
            prices=prices,
            signal=sig,
            eligibility=eligibility,
            rebalance="M",
            execution=spec["execution"],
            n_quantiles=N_QUANTILES,
            direction=DIRECTION,
            costs=spec["costs"],
        )
        summ = summarize(res, sector_map=sector_map)
        summ["meta"] = {
            k: v for k, v in res.meta.items() if k != "liquidations"
        }
        tw_dates = res.target_weights["date"]
        summ["first_rebalance_exec"] = (
            str(pd.Timestamp(tw_dates.min()).date()) if len(tw_dates) else None
        )
        summ["last_rebalance_exec"] = (
            str(pd.Timestamp(tw_dates.max()).date()) if len(tw_dates) else None
        )
        out[name] = summ
        if name == "base":
            base_result = res

    # Sign guard on real data: on the first executed rebalance the LONG leg
    # must hold the LOWEST signal-z names.
    tw = base_result.target_weights
    d0 = tw["date"].min()
    w0 = tw[tw["date"] == d0].set_index("ticker")["weight"]
    sig_w = sig.pivot(index="date", columns="ticker", values="value").sort_index()
    z0 = sig_w.loc[:d0].ffill().iloc[-1]
    z_long = float(z0[w0[w0 > 0].index].mean())
    z_short = float(z0[w0[w0 < 0].index].mean())
    assert z_long < z_short, (
        f"long leg mean z {z_long:.3f} must be below short leg {z_short:.3f}"
    )
    out["sign_guard"] = {
        "first_rebalance": str(pd.Timestamp(d0).date()),
        "long_leg_mean_z": z_long,
        "short_leg_mean_z": z_short,
    }

    # Base-cell extras ----------------------------------------------------- #
    net = base_result.daily_returns_net
    monthly = (1.0 + net).groupby(net.index.to_period("M")).prod() - 1.0
    lags = default_lags(21, 21, ic=monthly)
    out["base_monthly_hac"] = {
        "n_months": int(len(monthly)),
        "mean_monthly_net": float(monthly.mean()),
        "hac_lags": int(lags),
        "t_hac": float(hac_tstat(monthly, lags)),
        "construction": "monthly compounded net returns; lags via "
                        "default_lags(21, 21, ic=series)",
    }
    ff = ff_attribution(net)
    out["base_ff5_mom"] = {
        k: ff[k] for k in ("model", "n_days", "hac_lags", "alpha_ann",
                           "alpha_t_hac", "r2", "resid_vol_ann", "ir_resid")
    }
    out["base_ff5_mom"]["mkt_beta"] = ff["loadings"]["mkt_rf"]
    out["base_ff5_mom"]["loadings"] = ff["loadings"]
    out["base_ff5_mom"]["loadings_t_hac"] = ff["loadings_t_hac"]

    print(json.dumps(_jsonable(out), indent=2))


if __name__ == "__main__":
    main()
