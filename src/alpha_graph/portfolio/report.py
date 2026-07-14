"""Promotion-gate summary metrics for a BacktestResult.

Produces every metric named by reports/promotion_gate_c11.md. Conventions:

- Annualization: daily -> annual via sqrt(252) for Sharpe, x252 for means.
- Break-even cost (bps, PER SIDE): the flat per-side rate that zeroes the
  net arithmetic mean return, derived from gross mean and turnover only:
      1e4 * sum(gross daily returns) / (2 * sum(one-way turnover)).
  Borrow is excluded (it is not a per-trade cost); gate criterion 5 compares
  this number to the per-side trading-cost estimate.
- Beta: cov/var slope of NET daily returns on the equal-weight-universe
  daily return — a PROXY benchmark (survivor-biased universe, no dividend
  handling), computed inside the backtest.
- Year P&L shares use arithmetic sums of net daily returns (additive
  attribution); per-year returns are also reported compounded. The
  leave-one-year-out minimum share = 1 - max single-year share.
- Name/sector attribution uses per-name GROSS P&L (costs are not
  attributable to single names). Sector attribution needs `sector_map`
  (ticker -> sector); without it the sector flag is None and a note is
  emitted.
- Concentration flags (gate criterion 6): one year > 60% of net P&L, one
  sector > 50% of gross name P&L, top-10 names > 50% of gross name P&L.
  Shares are undefined when the relevant total is <= 0 (flags then None
  with a note — a strategy with non-positive total P&L has already failed
  the gate on other criteria).
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Mapping, Optional

import pandas as pd

from alpha_graph.portfolio.backtest import BacktestResult

TRADING_DAYS_PER_YEAR = 252


def _ann_sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=1)
    if not sd > 0:
        return float("nan")
    return float(daily.mean() / sd * math.sqrt(TRADING_DAYS_PER_YEAR))


def summarize(
    result: BacktestResult,
    by_year: bool = True,
    sector_map: Optional[Mapping[str, str]] = None,
) -> dict:
    """Promotion-gate metric summary of a backtest result (see module doc)."""
    g = result.daily_returns_gross
    n = result.daily_returns_net
    notes: list[str] = [
        "benchmark is an equal-weight-universe PROXY (survivor-biased, no dividends)",
        "short dividend cash flows are not modeled (see costs.py): net overstates the "
        "short leg where prices are not dividend-adjusted",
        "delisted names exit at last available close with no exit cost: optimistic vs "
        "true delisting returns",
    ]
    out: dict = {
        "start": result.meta.get("start"),
        "end": result.meta.get("end"),
        "n_days": int(len(n)),
        "cost_model": asdict(result.cost_model),
        "n_delistings": result.n_delistings,
        "skipped_rebalances": result.skipped_rebalances,
    }

    # Sharpe / return / drawdown ------------------------------------------- #
    out["sharpe_gross_ann"] = _ann_sharpe(g)
    out["sharpe_net_ann"] = _ann_sharpe(n)
    out["ann_return_gross"] = float(g.mean() * TRADING_DAYS_PER_YEAR)
    out["ann_return_net"] = float(n.mean() * TRADING_DAYS_PER_YEAR)
    nav = (1.0 + n).cumprod()
    out["max_drawdown_net"] = float((nav / nav.cummax() - 1.0).min())

    # Turnover and break-even ---------------------------------------------- #
    years = len(n) / TRADING_DAYS_PER_YEAR
    total_turn = float(result.turnover.sum())
    out["turnover_oneway_ann"] = total_turn / years if years > 0 else float("nan")
    if total_turn > 0:
        out["breakeven_cost_bps"] = float(1e4 * g.sum() / (2.0 * total_turn))
    else:
        out["breakeven_cost_bps"] = float("nan")
        notes.append("zero turnover: break-even cost undefined")

    # Beta vs equal-weight universe ---------------------------------------- #
    pair = pd.concat([n, result.benchmark_returns], axis=1, join="inner").dropna()
    if len(pair) >= 3 and pair.iloc[:, 1].var(ddof=1) > 0:
        out["beta_vs_ew_universe"] = float(
            pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var(ddof=1)
        )
    else:
        out["beta_vs_ew_universe"] = float("nan")
        notes.append("benchmark variance ~0 or too few days: beta undefined")

    # Leg decomposition (gate criterion 7) ---------------------------------- #
    long_total = float(result.long_returns.sum())
    short_total = float(result.short_returns.sum())
    out["long_leg_total_return"] = long_total
    out["short_leg_total_return"] = short_total
    denom = long_total + short_total
    if abs(denom) > 1e-15:
        out["long_leg_share"] = long_total / denom
        out["short_leg_share"] = short_total / denom
    else:
        out["long_leg_share"] = float("nan")
        out["short_leg_share"] = float("nan")

    # Per-year P&L, LOYO, year concentration -------------------------------- #
    by_yr_sum = n.groupby(n.index.year).sum()
    total_net = float(n.sum())
    if by_year:
        out["net_by_year"] = {
            int(y): float((1.0 + n[n.index.year == y]).prod() - 1.0)
            for y in by_yr_sum.index
        }
    flag_year: Optional[bool]
    if total_net > 0:
        shares = by_yr_sum / total_net
        out["max_year_share"] = float(shares.max())
        out["max_year"] = int(shares.idxmax())
        out["loyo_min_share"] = float(1.0 - shares.max())
        flag_year = bool(shares.max() > 0.60)
    else:
        out["max_year_share"] = float("nan")
        out["max_year"] = None
        out["loyo_min_share"] = float("nan")
        flag_year = None
        notes.append("total net P&L <= 0: year-concentration shares undefined")

    # Name concentration ----------------------------------------------------- #
    pnl = result.name_pnl.dropna()
    total_name = float(pnl.sum())
    flag_top10: Optional[bool]
    if total_name > 0:
        top10 = pnl.nlargest(10)
        out["top10_name_share"] = float(top10.sum() / total_name)
        out["top10_names"] = list(top10.index)
        flag_top10 = bool(out["top10_name_share"] > 0.50)
    else:
        out["top10_name_share"] = float("nan")
        out["top10_names"] = []
        flag_top10 = None
        notes.append("total gross name P&L <= 0: top-10 share undefined")

    # Sector concentration ---------------------------------------------------- #
    flag_sector: Optional[bool]
    if sector_map is not None and total_name > 0:
        sectors = pnl.index.map(lambda t: sector_map.get(t, "UNMAPPED"))
        sector_pnl = pnl.groupby(sectors).sum()
        out["sector_pnl_share"] = {
            str(s): float(v / total_name) for s, v in sector_pnl.items()
        }
        out["max_sector_share"] = float(sector_pnl.max() / total_name)
        out["max_sector"] = str(sector_pnl.idxmax())
        flag_sector = bool(out["max_sector_share"] > 0.50)
    else:
        out["sector_pnl_share"] = None
        out["max_sector_share"] = float("nan")
        out["max_sector"] = None
        flag_sector = None
        if sector_map is None:
            notes.append("no sector_map provided: sector concentration skipped")

    out["flags"] = {
        "one_year_gt_60pct": flag_year,
        "one_sector_gt_50pct": flag_sector,
        "top10_names_gt_50pct": flag_top10,
    }
    out["notes"] = notes
    return out
