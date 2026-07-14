"""Portfolio construction and execution-cost layer.

Converts a daily cross-sectional signal into net-of-cost long-short P&L:
quantile weight construction, a share-based daily backtest, a transparent
cost model, and a promotion-gate metric summary
(see reports/promotion_gate_c11.md).
"""

from alpha_graph.portfolio.backtest import BacktestResult, run_ls_backtest
from alpha_graph.portfolio.construction import cap_and_redistribute, quantile_ls_weights
from alpha_graph.portfolio.costs import CostModel
from alpha_graph.portfolio.report import summarize

__all__ = [
    "BacktestResult",
    "CostModel",
    "cap_and_redistribute",
    "quantile_ls_weights",
    "run_ls_backtest",
    "summarize",
]
