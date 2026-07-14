"""Share-based daily long-short backtest.

Semantics (documented choices — read before interpreting any output)
--------------------------------------------------------------------
- The signal frame is ALREADY availability-dated by the caller. At each
  rebalance decision date t the engine uses, per ticker, the latest non-NaN
  signal value with date <= t (as-of join). No further lagging happens here.
- Optional `eligibility` is a DAILY state frame. When supplied, a name can
  enter a target only when it is explicitly eligible on the decision date.
  Signal carry is reset across ineligible spells, so an index removal cannot
  leak a stale score into a later rebalance or re-entry.
- execution="close": trade at the close of the decision date t.
  execution="next_open": trade at the open of the next trading day after t
  (the promotion gate's "next tradable window"; requires an `open` column
  in `prices`). With "close", the day-t return of a new position is 0 (it is
  established at that close); with "next_open", the execution day's return
  is old book close->open plus new book open->close.
- Between rebalances SHARES are held fixed: weights drift with returns.
  There is no daily weight renormalization.
- A rebalance TRADES only if the new target weight vector differs from the
  previous target vector. A static signal therefore produces zero turnover
  after initiation: weights keep drifting and are NOT traded back to equal
  weight. Any rank change re-trades the full target (which also resets
  drift).
- Delistings: our price data is survivor-biased and has no delisting
  returns. A held name whose prices stop mid-sample is liquidated at its
  LAST AVAILABLE CLOSE on the first trading day after it; proceeds sit in
  cash inside the book until the next rebalance redeploys them. This is
  OPTIMISTIC versus reality (no delisting-loss return, and the forced exit
  is charged no trading cost). Events are counted in the result and logged.
- Turnover convention: one-way turnover = 0.5 * sum(|w_new - w_drifted|)
  at each executed rebalance (the standard sum|dw|/2 convention;
  initiation of a gross-2.0 book = 1.0).
- Costs are RETURN DRAG on the gross series; positions are sized on the
  GROSS NAV path:
      net_r(d) = gross_r(d)
                 - traded_fraction(d) * costs.trading_cost_rate
                 - short_gross_weight(d-1) * costs.daily_borrow_rate
  where traded_fraction = sum|w_new - w_drifted| (2 x one-way turnover) as a
  fraction of NAV at the trade, and short_gross_weight is measured at the
  prior close (borrow is charged for each overnight a short is held).
  Second-order denominator mixing (rate x daily move) is ignored. Short
  dividend cash flows are NOT modeled — see costs.py.
- Benchmark: equal-weight mean of same-day close-to-close returns over all
  names with valid prices that day. This is a PROXY benchmark
  (survivor-biased universe, no dividend/cash handling) used for the
  report's beta.
- Rebalances where the valid cross-section has fewer than n_quantiles names
  are SKIPPED (previous book held) and counted in `skipped_rebalances` —
  EXCEPT when explicit `eligibility` is supplied and a book is held: then
  the book is force-liquidated to cash at real prices with trading costs
  (fail-closed; counted in `n_forced_cash_insufficient_eligibility`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.portfolio.construction import quantile_ls_weights
from alpha_graph.portfolio.costs import CostModel


@dataclass
class BacktestResult:
    """Output of run_ls_backtest. All return series are date-indexed daily."""

    daily_returns_gross: pd.Series
    daily_returns_net: pd.Series
    long_returns: pd.Series          # long-leg contribution to gross return (on total NAV)
    short_returns: pd.Series         # short-leg contribution to gross return (on total NAV)
    turnover: pd.Series              # one-way (sum|dw|/2), indexed by execution dates
    trading_costs: pd.Series         # daily trading-cost drag (fraction of NAV)
    borrow_costs: pd.Series          # daily borrow drag (fraction of NAV)
    benchmark_returns: pd.Series     # equal-weight-universe daily return (proxy)
    target_weights: pd.DataFrame     # executed targets, long format: date, ticker, weight
    name_pnl: pd.Series              # cumulative gross P&L per ticker (units of initial NAV)
    cost_model: CostModel
    n_delistings: int
    delisted: list[str]
    skipped_rebalances: int
    meta: dict = field(default_factory=dict)


def _pivot(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = frame.pivot(index="date", columns="ticker", values=value_col).sort_index()
    return out


def _decision_dates(dates: pd.DatetimeIndex, sig_start: pd.Timestamp,
                    rebalance: Union[str, int]) -> pd.DatetimeIndex:
    elig = dates[dates >= sig_start]
    if len(elig) == 0:
        raise ValueError("no trading dates on/after the first signal date")
    if isinstance(rebalance, bool):
        raise ValueError("rebalance must be 'M', 'W', or a positive int")
    if isinstance(rebalance, (int, np.integer)):
        if rebalance < 1:
            raise ValueError(f"int rebalance must be >= 1, got {rebalance}")
        return elig[:: int(rebalance)]
    if rebalance == "M":
        keys = elig.to_period("M")
    elif rebalance == "W":
        keys = elig.to_period("W")  # W-SUN periods: last trading day per calendar week
    else:
        raise ValueError(f"rebalance must be 'M', 'W', or a positive int, got {rebalance!r}")
    last = pd.Series(elig, index=keys).groupby(level=0).max()
    return pd.DatetimeIndex(last.to_numpy()).sort_values()


def run_ls_backtest(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    rebalance: Union[str, int] = "M",
    execution: str = "close",
    n_quantiles: int = 10,
    direction: int = 1,
    costs: CostModel | None = None,
    max_weight: float = 0.05,
    eligibility: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run a decile-style long-short backtest with share-based accounting.

    prices: long frame with columns ticker, date, close (+ open for
    execution="next_open"). signal: long frame with columns ticker, date,
    value — already availability-dated by the caller. eligibility: optional
    daily long frame with ticker/date and an optional boolean `eligible`
    column; row presence means eligible when that column is omitted. Missing
    ticker-dates fail closed. See module docstring for all semantics.
    """
    costs = costs if costs is not None else CostModel()
    if not isinstance(costs, CostModel):
        raise TypeError("costs must be a CostModel")
    if execution not in ("close", "next_open"):
        raise ValueError(f"execution must be 'close' or 'next_open', got {execution!r}")
    for col in ("ticker", "date", "close"):
        if col not in prices.columns:
            raise ValueError(f"prices is missing column {col!r}")
    if execution == "next_open" and "open" not in prices.columns:
        raise ValueError("execution='next_open' requires an 'open' column in prices")
    for col in ("ticker", "date", "value"):
        if col not in signal.columns:
            raise ValueError(f"signal is missing column {col!r}")
    if eligibility is not None:
        for col in ("ticker", "date"):
            if col not in eligibility.columns:
                raise ValueError(f"eligibility is missing column {col!r}")

    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    close_w = _pivot(px, "close").dropna(axis=1, how="all")
    dates = close_w.index
    tickers = close_w.columns.to_numpy()
    n_tk = len(tickers)
    col_of = {t: i for i, t in enumerate(tickers)}

    notna_c = close_w.notna().to_numpy()
    C = close_w.ffill().to_numpy(dtype=float)       # stale-marked after death, NaN before birth
    last_row = len(dates) - 1 - notna_c[::-1].argmax(axis=0)  # last real price row per ticker

    use_open = execution == "next_open"
    if use_open:
        open_w = _pivot(px, "open").reindex(columns=close_w.columns)
        notna_o = open_w.notna().to_numpy()
        O = open_w.ffill().to_numpy(dtype=float)

    sg = signal.copy()
    sg["date"] = pd.to_datetime(sg["date"])
    sig_w = _pivot(sg, "value")
    if sig_w.empty:
        raise ValueError("signal frame is empty")
    union = dates.union(sig_w.index)
    sig_union = sig_w.reindex(union)

    elig_w: pd.DataFrame | None = None
    if eligibility is not None:
        eg = eligibility.copy()
        eg["date"] = pd.to_datetime(eg["date"])
        if eg.duplicated(["ticker", "date"]).any():
            raise ValueError("eligibility has duplicate ticker/date rows")
        if "eligible" not in eg.columns:
            eg["eligible"] = True
        values = set(eg["eligible"].dropna().unique())
        if not values.issubset({True, False, 0, 1}):
            raise ValueError("eligibility.eligible must contain only booleans")
        # Off-calendar guard (review F1): a signal row stamped on a date the
        # eligibility frame does not cover AT ALL (weekend/holiday filing —
        # 636 Form 4 dates in-range are non-trading days) must inherit each
        # name's as-of eligibility state, NOT a synthetic all-False row that
        # would reset every ticker's carry spell and force-liquidate the
        # book. On the frame's own dates the contract is unchanged: an
        # absent ticker-date fails closed.
        raw = _pivot(eg, "eligible").reindex(columns=sig_union.columns)
        on_grid = raw.reindex(index=union)
        off_calendar = ~union.isin(raw.index)
        elig_w = on_grid.fillna(False)
        if off_calendar.any():
            elig_w.loc[off_calendar] = on_grid.ffill().loc[off_calendar]
        elig_w = elig_w.fillna(False).astype(bool)

        # Carry sparse signals only within one continuous eligible spell.
        sig_asof = pd.DataFrame(index=union, columns=sig_union.columns, dtype=float)
        for ticker in sig_union.columns:
            eligible_t = elig_w[ticker]
            spell = (~eligible_t).cumsum()
            sig_asof[ticker] = (
                sig_union[ticker]
                .where(eligible_t)
                .groupby(spell)
                .ffill()
                .where(eligible_t)
            )
    else:
        sig_asof = sig_union.ffill()                 # latest non-NaN value per ticker, as-of

    decisions = _decision_dates(dates, sig_w.index.min(), rebalance)
    pos = {d: i for i, d in enumerate(dates)}
    exec_map: dict[int, pd.Timestamp] = {}           # execution row -> decision date
    dropped_tail = 0
    for t in decisions:
        r = pos[t] + 1 if use_open else pos[t]
        if r >= len(dates):
            dropped_tail += 1                        # next_open decision on the last date
            continue
        exec_map[r] = t

    if not exec_map:
        raise ValueError("no executable rebalance dates")

    # --- state -------------------------------------------------------------
    h = np.zeros(n_tk)                               # shares
    nav = 1.0
    sgw_prev = 0.0                                   # short gross weight at prior close
    prev_target: np.ndarray | None = None
    started = False

    rec_dates: list[pd.Timestamp] = []
    rec_gross: list[float] = []
    rec_net: list[float] = []
    rec_long: list[float] = []
    rec_short: list[float] = []
    rec_tc: list[float] = []
    rec_bc: list[float] = []
    turn_dates: list[pd.Timestamp] = []
    turn_vals: list[float] = []
    tw_rows: list[tuple[pd.Timestamp, str, float]] = []
    name_pnl = np.zeros(n_tk)
    liquidations: list[tuple[pd.Timestamp, str]] = []
    skipped_small = 0
    n_trades = 0
    n_skip_unchanged = 0
    n_forced_cash = 0

    def _contrib(hold: np.ndarray, dp: np.ndarray) -> np.ndarray:
        out = np.where(hold != 0.0, hold * dp, 0.0)
        if np.isnan(out).any():
            raise RuntimeError("NaN P&L on a held name — internal accounting bug")
        return out

    def _form_target(decision_date: pd.Timestamp, exec_row: int) -> np.ndarray | None:
        """Full-vector target weights, or None if the cross-section is too small."""
        nonlocal n_forced_cash, skipped_small
        row = sig_asof.loc[decision_date]
        scores = row.dropna()
        tradeable = notna_c[exec_row]
        if use_open:
            tradeable = tradeable & notna_o[exec_row]
        if elig_w is not None:
            eligible_today = elig_w.loc[decision_date].reindex(tickers, fill_value=False)
            tradeable = tradeable & eligible_today.to_numpy(dtype=bool)
        keep = [t for t in scores.index if t in col_of and tradeable[col_of[t]]]
        if len(keep) < n_quantiles:
            skipped_small += 1
            if elig_w is not None and started:
                if np.any(h != 0.0):
                    n_forced_cash += 1
                    logger.warning(
                        "rebalance {} has {} eligible names < n_quantiles={}; "
                        "liquidating the existing book to cash",
                        decision_date.date(), len(keep), n_quantiles,
                    )
                return np.zeros(n_tk)
            logger.warning(
                "rebalance {} skipped: {} valid names < n_quantiles={}",
                decision_date.date(), len(keep), n_quantiles,
            )
            return None
        w = quantile_ls_weights(scores.loc[keep], n_quantiles=n_quantiles,
                                direction=direction, max_weight=max_weight)
        full = np.zeros(n_tk)
        for t, v in w.items():
            full[col_of[t]] = v
        return full

    first_exec_row = min(exec_map)
    for di in range(first_exec_row, len(dates)):
        d = dates[di]
        exec_here = di in exec_map

        gross_pnl = 0.0
        long_c = 0.0
        short_c = 0.0
        tc_drag = 0.0
        nav_prev = nav

        # 0. liquidate names whose prices have stopped (first day past the last
        #    print): the position converts to cash at the last available close
        #    (already the NAV mark), so it contributes nothing from here on.
        dead = np.where((h != 0.0) & (di > last_row))[0]
        for j in dead:
            liquidations.append((d, str(tickers[j])))
            h[j] = 0.0

        # 1. form targets on execution days (as-of the decision date's signal)
        new_target = _form_target(exec_map[di], di) if exec_here else None
        will_trade = (
            new_target is not None
            and (prev_target is None or not np.array_equal(new_target, prev_target))
        )
        traded_at_open = False

        # 2. accrue P&L close(d-1) -> close(d) on held shares (or the pre-open
        #    piece close(d-1) -> open(d) when trading at d's open below).
        if started:
            if use_open and will_trade:
                dp_pre = O[di] - C[di - 1]
            else:
                dp_pre = C[di] - C[di - 1]
            c_pre = _contrib(h, dp_pre)
            gross_pnl += float(c_pre.sum())
            long_c += float(c_pre[h > 0].sum())
            short_c += float(c_pre[h < 0].sum())
            name_pnl += c_pre

        borrow_drag = sgw_prev * costs.daily_borrow_rate if started else 0.0

        # 3. rebalance
        if new_target is not None:
            if will_trade:
                exec_px = O[di] if use_open else C[di]
                nav_at_trade = nav_prev + gross_pnl        # NAV at the execution point
                held_val = np.where(h != 0.0, h * exec_px, 0.0)
                w_drift = held_val / nav_at_trade
                traded = float(np.abs(new_target - w_drift).sum())
                turn_dates.append(d)
                turn_vals.append(traded / 2.0)
                tc_drag = traded * costs.trading_cost_rate
                with np.errstate(invalid="ignore", divide="ignore"):
                    h = np.where(new_target != 0.0, new_target * nav_at_trade / exec_px, 0.0)
                for j in np.nonzero(new_target)[0]:
                    tw_rows.append((d, str(tickers[j]), float(new_target[j])))
                prev_target = new_target
                n_trades += 1
                traded_at_open = use_open
                started = True
            else:
                turn_dates.append(d)
                turn_vals.append(0.0)
                n_skip_unchanged += 1

        # 4. post-open piece open(d) -> close(d) on the NEW book, if traded at open
        if traded_at_open:
            dp_post = C[di] - O[di]
            c_post = _contrib(h, dp_post)
            gross_pnl += float(c_post.sum())
            long_c += float(c_post[h > 0].sum())
            short_c += float(c_post[h < 0].sum())
            name_pnl += c_post

        if not started:
            continue

        r_gross = gross_pnl / nav_prev
        nav = nav_prev + gross_pnl
        r_net = r_gross - tc_drag - borrow_drag

        rec_dates.append(d)
        rec_gross.append(r_gross)
        rec_net.append(r_net)
        rec_long.append(long_c / nav_prev)
        rec_short.append(short_c / nav_prev)
        rec_tc.append(tc_drag)
        rec_bc.append(borrow_drag)

        short_val = np.where(h < 0, h * C[di], 0.0)
        sgw_prev = float(-short_val.sum()) / nav

    if not rec_dates:
        raise ValueError("backtest never initiated: every rebalance was skipped")

    idx = pd.DatetimeIndex(rec_dates, name="date")
    bench = (
        close_w.pct_change(fill_method=None).mean(axis=1).reindex(idx)
    )
    result = BacktestResult(
        daily_returns_gross=pd.Series(rec_gross, index=idx, name="gross"),
        daily_returns_net=pd.Series(rec_net, index=idx, name="net"),
        long_returns=pd.Series(rec_long, index=idx, name="long"),
        short_returns=pd.Series(rec_short, index=idx, name="short"),
        turnover=pd.Series(turn_vals, index=pd.DatetimeIndex(turn_dates, name="date"),
                           name="oneway_turnover"),
        trading_costs=pd.Series(rec_tc, index=idx, name="trading_cost"),
        borrow_costs=pd.Series(rec_bc, index=idx, name="borrow_cost"),
        benchmark_returns=bench.rename("ew_universe"),
        target_weights=pd.DataFrame(tw_rows, columns=["date", "ticker", "weight"]),
        name_pnl=pd.Series(name_pnl, index=pd.Index(tickers, name="ticker"),
                           name="gross_pnl"),
        cost_model=costs,
        n_delistings=len(liquidations),
        delisted=sorted({t for _, t in liquidations}),
        skipped_rebalances=skipped_small,
        meta={
            "rebalance": rebalance,
            "execution": execution,
            "n_quantiles": n_quantiles,
            "direction": direction,
            "max_weight": max_weight,
            "n_rebalance_decisions": len(decisions),
            "n_executed_trades": n_trades,
            "n_skipped_unchanged_target": n_skip_unchanged,
            "n_skipped_small_cross_section": skipped_small,
            "n_forced_cash_insufficient_eligibility": n_forced_cash,
            "n_decisions_dropped_no_next_open": dropped_tail,
            "explicit_eligibility": eligibility is not None,
            "liquidations": liquidations,
            "start": idx[0],
            "end": idx[-1],
        },
    )
    if liquidations:
        logger.warning(
            "{} delisting liquidation(s) at last available close (optimistic vs true "
            "delisting returns; no exit cost charged): {}",
            len(liquidations), result.delisted,
        )
    logger.info(
        "L/S backtest {} -> {}: {} days, {} trades, {} unchanged-target skips, "
        "{} small-cross-section skips, {} delistings",
        idx[0].date(), idx[-1].date(), len(idx), n_trades, n_skip_unchanged,
        skipped_small, len(liquidations),
    )
    return result
