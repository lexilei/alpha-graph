"""Execution-cost model for the long-short backtest.

Cost semantics (applied as return drag on the gross daily series; see
backtest.py for exactly where each term lands):

- Trading: at each executed rebalance,
  drag = traded notional fraction of NAV (sum |w_new - w_drifted|, i.e.
  2 x one-way turnover) x (half_spread_bps + commission_bps) / 1e4.
  The rate is PER SIDE of each trade: every unit of notional bought or sold
  pays half-spread + commission once.
- Borrow: accrues daily as
  short-leg gross weight (fraction of NAV at the prior close)
  x borrow_bps_pa / 1e4 / 252, charged for each day a short position is
  held overnight.
- `doubled=True` multiplies every component x2 — the promotion gate's
  cost-doubling robustness cell (reports/promotion_gate_c11.md, criterion 4).

Known limitation — SHORT DIVIDEND CASH FLOWS ARE NOT MODELED. There is no
dividend series in the data. The cached prices are yfinance auto-adjusted
closes, so dividend effects enter price returns approximately (an adjusted
short "pays" the dividend through the adjusted price path), but explicit
payment-date cash flows, borrow-vs-dividend timing, and any unadjusted price
feed are NOT handled. To the extent adjustment is absent or imperfect, net
results OVERSTATE short-leg P&L by up to the dividend yield of shorted names
times short gross exposure. Gate criterion 2 lists short dividends as a
required cost: any gate reading from this model must carry this caveat.
"""

from __future__ import annotations

from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class CostModel:
    """Flat-rate cost model: half-spread + commission per side, borrow p.a."""

    half_spread_bps: float = 0.0
    commission_bps: float = 0.0
    borrow_bps_pa: float = 0.0
    doubled: bool = False

    def __post_init__(self) -> None:
        for field in ("half_spread_bps", "commission_bps", "borrow_bps_pa"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be >= 0")

    @property
    def multiplier(self) -> float:
        """x2 when doubled (gate robustness cell), else x1."""
        return 2.0 if self.doubled else 1.0

    @property
    def trading_cost_rate(self) -> float:
        """Cost per unit of traded notional (per side), as a fraction."""
        return (self.half_spread_bps + self.commission_bps) / 1e4 * self.multiplier

    @property
    def daily_borrow_rate(self) -> float:
        """Daily borrow cost per unit of short gross notional (bps_pa / 252)."""
        return self.borrow_bps_pa / 1e4 / TRADING_DAYS_PER_YEAR * self.multiplier
