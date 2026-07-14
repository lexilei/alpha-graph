"""Tests for alpha_graph.portfolio — synthetic data only.

Covers construction (quantile L/S weights, cap redistribution), the cost
model, the share-based backtest engine, and the promotion-gate summary.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.portfolio import (
    CostModel,
    cap_and_redistribute,
    quantile_ls_weights,
    run_ls_backtest,
    summarize,
)

TOL = 1e-12


def _prices_long(dates, close: dict, open_: dict | None = None) -> pd.DataFrame:
    """Long price frame from per-ticker lists; None entries = no row (delisted)."""
    rows = []
    for t, cs in close.items():
        os_ = open_[t] if open_ is not None else [None] * len(cs)
        for d, c, o in zip(dates, cs, os_):
            if c is None:
                continue
            row = {"ticker": t, "date": d, "close": float(c)}
            if open_ is not None:
                row["open"] = float(o)
            rows.append(row)
    return pd.DataFrame(rows)


def _signal_long(rows: list) -> pd.DataFrame:
    """Signal frame from [(date, {ticker: value})] rows."""
    recs = [
        {"ticker": t, "date": d, "value": v}
        for d, mapping in rows
        for t, v in mapping.items()
    ]
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #

def test_quantile_weights_basic_and_nan_exclusion():
    scores = pd.Series(
        {f"N{i:02d}": float(i) for i in range(10)} | {"NAN1": np.nan}
    )
    w = quantile_ls_weights(scores, n_quantiles=5, direction=1, max_weight=0.5)
    # 10 valid names, 5 quantiles -> 2 per bin; top {N08,N09}, bottom {N00,N01}
    assert set(w.index) == {"N08", "N09", "N00", "N01"}
    assert w["N08"] == pytest.approx(0.5, abs=TOL)
    assert w["N09"] == pytest.approx(0.5, abs=TOL)
    assert w["N00"] == pytest.approx(-0.5, abs=TOL)
    assert w["N01"] == pytest.approx(-0.5, abs=TOL)
    assert abs(w.sum()) < TOL                      # dollar neutral
    assert abs(w.abs().sum() - 2.0) < TOL          # gross 2.0
    assert "NAN1" not in w.index


def test_quantile_weights_direction_flip():
    scores = pd.Series({f"N{i}": float(i) for i in range(8)})
    w_pos = quantile_ls_weights(scores, n_quantiles=4, direction=1, max_weight=0.5)
    w_neg = quantile_ls_weights(scores, n_quantiles=4, direction=-1, max_weight=0.5)
    assert ((w_neg + w_pos).abs() < TOL).all()     # exact sign flip


def test_quantile_weights_deterministic_ties():
    # Five names tied at the top score; with 10 names / 5 quantiles the top
    # bin takes 2 names — the tie must break alphabetically by ticker.
    scores = pd.Series(
        {"A": 9.0, "B": 9.0, "C": 9.0, "D": 9.0, "E": 9.0,
         "F": 1.0, "G": 2.0, "H": 3.0, "I": 4.0, "J": 5.0}
    )
    w1 = quantile_ls_weights(scores, n_quantiles=5, max_weight=0.5)
    w2 = quantile_ls_weights(scores.sample(frac=1.0, random_state=7), n_quantiles=5,
                             max_weight=0.5)
    # ascending sort: F,G,H,I,J then A,B,C,D,E -> top-2 = {D, E}; bottom-2 = {F, G}
    assert set(w1[w1 > 0].index) == {"D", "E"}
    assert set(w1[w1 < 0].index) == {"F", "G"}
    pd.testing.assert_series_equal(w1, w2)         # input order irrelevant


def test_quantile_weights_validation():
    scores = pd.Series({"A": 1.0, "B": 2.0, "C": np.nan})
    with pytest.raises(ValueError):                # 2 valid < 3 quantiles
        quantile_ls_weights(scores, n_quantiles=3)
    with pytest.raises(ValueError):
        quantile_ls_weights(scores, n_quantiles=2, direction=0)
    with pytest.raises(ValueError):                # infeasible cap: 1 name/leg at 0.05
        quantile_ls_weights(pd.Series({"A": 1.0, "B": 2.0}), n_quantiles=2,
                            max_weight=0.05)


def test_cap_and_redistribute():
    # one round: 0.5 capped at 0.4, excess pro-rata to the others
    w = cap_and_redistribute(pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}), cap=0.4)
    assert w["A"] == pytest.approx(0.4, abs=TOL)
    assert w["B"] == pytest.approx(0.36, abs=TOL)
    assert w["C"] == pytest.approx(0.24, abs=TOL)
    assert w.sum() == pytest.approx(1.0, abs=TOL)
    # two rounds: redistribution pushes B over the cap, second pass fixes it
    w2 = cap_and_redistribute(pd.Series({"A": 0.6, "B": 0.35, "C": 0.05}), cap=0.4)
    assert w2["A"] == pytest.approx(0.4, abs=TOL)
    assert w2["B"] == pytest.approx(0.4, abs=TOL)
    assert w2["C"] == pytest.approx(0.2, abs=TOL)
    assert w2.sum() == pytest.approx(1.0, abs=TOL)
    assert (w2 <= 0.4 + 1e-9).all()
    # infeasible
    with pytest.raises(ValueError):
        cap_and_redistribute(pd.Series({"A": 0.5, "B": 0.5}), cap=0.3)
    # exact fit is feasible
    w3 = cap_and_redistribute(pd.Series({"A": 0.7, "B": 0.3}), cap=0.5)
    assert w3["A"] == pytest.approx(0.5, abs=TOL)
    assert w3["B"] == pytest.approx(0.5, abs=TOL)


# --------------------------------------------------------------------------- #
# costs
# --------------------------------------------------------------------------- #

def test_cost_model_rates():
    cm = CostModel(half_spread_bps=10.0, commission_bps=5.0, borrow_bps_pa=252.0)
    assert cm.trading_cost_rate == pytest.approx(0.0015, abs=TOL)
    assert cm.daily_borrow_rate == pytest.approx(1e-4, abs=TOL)
    cm2 = CostModel(half_spread_bps=10.0, commission_bps=5.0, borrow_bps_pa=252.0,
                    doubled=True)
    assert cm2.trading_cost_rate == pytest.approx(0.0030, abs=TOL)
    assert cm2.daily_borrow_rate == pytest.approx(2e-4, abs=TOL)
    with pytest.raises(ValueError):
        CostModel(half_spread_bps=-1.0)


# --------------------------------------------------------------------------- #
# backtest — hand-computed exactness
# --------------------------------------------------------------------------- #

def test_hand_computed_two_period_four_names():
    """2-period, 4-name case, every number derived by hand; asserted to 1e-12.

    d0: signal A4,B3,C2,D1 -> long {A,B} +0.5, short {C,D} -0.5; trade at close.
    d1: A +10%, C -5%, D -10% -> gross 0.125; new signal A4,C3,B2,D1 -> long
        {A,C}, short {B,D}; turnover (sum|dw|)/2 = 89/90.
    d2: A +10%, D -10% on new book -> gross 0.1; as-of signal unchanged ->
        identical target -> no trade, zero turnover.
    Costs 15 bps/side flat: drag = traded_fraction x 0.0015.
    """
    dates = list(pd.bdate_range("2020-01-06", periods=3))
    prices = _prices_long(dates, {
        "A": [100, 110, 121],
        "B": [100, 100, 100],
        "C": [100, 95, 95],
        "D": [100, 90, 81],
    })
    signal = _signal_long([
        (dates[0], {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
        (dates[1], {"A": 4.0, "C": 3.0, "B": 2.0, "D": 1.0}),
    ])
    cm = CostModel(half_spread_bps=10.0, commission_bps=5.0)
    res = run_ls_backtest(prices, signal, rebalance=1, execution="close",
                          n_quantiles=2, direction=1, costs=cm, max_weight=0.5)

    np.testing.assert_allclose(res.daily_returns_gross.to_numpy(),
                               [0.0, 0.125, 0.1], rtol=0, atol=TOL)
    np.testing.assert_allclose(res.turnover.to_numpy(),
                               [1.0, 89.0 / 90.0, 0.0], rtol=0, atol=TOL)
    assert list(res.turnover.index) == dates
    np.testing.assert_allclose(res.trading_costs.to_numpy(),
                               [0.003, (178.0 / 90.0) * 0.0015, 0.0],
                               rtol=0, atol=TOL)
    np.testing.assert_allclose(
        res.daily_returns_net.to_numpy(),
        [-0.003, 0.125 - (178.0 / 90.0) * 0.0015, 0.1], rtol=0, atol=TOL)
    np.testing.assert_allclose(res.long_returns.to_numpy(),
                               [0.0, 0.05, 0.05], rtol=0, atol=TOL)
    np.testing.assert_allclose(res.short_returns.to_numpy(),
                               [0.0, 0.075, 0.05], rtol=0, atol=TOL)
    # exact weights at both executed rebalances
    tw = res.target_weights.set_index(["date", "ticker"])["weight"]
    exp_d0 = {"A": 0.5, "B": 0.5, "C": -0.5, "D": -0.5}
    exp_d1 = {"A": 0.5, "C": 0.5, "B": -0.5, "D": -0.5}
    for tkr, w in exp_d0.items():
        assert abs(tw.loc[(dates[0], tkr)] - w) < TOL
    for tkr, w in exp_d1.items():
        assert abs(tw.loc[(dates[1], tkr)] - w) < TOL
    assert set(res.target_weights["date"].unique()) == {dates[0], dates[1]}
    # per-name gross P&L and NAV identity
    exp_pnl = {"A": 0.10625, "B": 0.0, "C": 0.025, "D": 0.10625}
    for tkr, v in exp_pnl.items():
        assert abs(res.name_pnl[tkr] - v) < TOL
    assert abs((1.0 + res.daily_returns_gross).prod() - 1.2375) < TOL
    assert res.meta["n_executed_trades"] == 2
    assert res.meta["n_skipped_unchanged_target"] == 1
    assert res.n_delistings == 0


def test_deterministic_outperformance_and_direction_flip():
    """Top decile +1%/day, bottom -1%/day -> L/S ~2%/day gross; -1 flips sign.

    Exactly 2% the day after formation; later days decay mildly because shares
    are held fixed (no re-trade on a static signal) while NAV compounds.
    """
    n_days = 10
    dates = list(pd.bdate_range("2020-01-06", periods=n_days))
    close = {}
    for i in range(20):
        t = f"N{i:02d}"
        if i >= 18:
            close[t] = [100.0 * 1.01 ** k for k in range(n_days)]
        elif i < 2:
            close[t] = [100.0 * 0.99 ** k for k in range(n_days)]
        else:
            close[t] = [100.0] * n_days
    prices = _prices_long(dates, close)
    signal = _signal_long([(dates[0], {f"N{i:02d}": float(i) for i in range(20)})])

    res = run_ls_backtest(prices, signal, rebalance=5, execution="close",
                          n_quantiles=10, direction=1, max_weight=0.5)
    g = res.daily_returns_gross
    assert abs(g.iloc[0]) < TOL                       # initiated at first close
    assert abs(g.iloc[1] - 0.02) < 1e-9               # 1% long + 1% short
    assert 0.017 < g.iloc[1:].mean() < 0.021
    assert (g.iloc[1:] > 0.016).all()
    np.testing.assert_allclose(res.turnover.to_numpy(), [1.0, 0.0],
                               rtol=0, atol=TOL)      # static signal: no re-trade

    res_neg = run_ls_backtest(prices, signal, rebalance=5, execution="close",
                              n_quantiles=10, direction=-1, max_weight=0.5)
    gn = res_neg.daily_returns_gross
    assert abs(gn.iloc[1] + 0.02) < 1e-9
    assert (gn.iloc[1:] < 0).all()


def test_static_signal_zero_turnover_weekly():
    """Static signal: weights drift with prices but no trades after initiation."""
    n_days = 20
    dates = list(pd.bdate_range("2020-01-06", periods=n_days))
    close = {
        "A": [100.0 * 1.01 ** k for k in range(n_days)],
        "B": [100.0 * 0.995 ** k for k in range(n_days)],
        "C": [100.0 + 1.0 * k for k in range(n_days)],
        "D": [100.0 - 0.5 * k for k in range(n_days)],
    }
    prices = _prices_long(dates, close)
    signal = _signal_long([(dates[0], {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})])
    cm = CostModel(half_spread_bps=10.0)
    res = run_ls_backtest(prices, signal, rebalance="W", execution="close",
                          n_quantiles=2, costs=cm, max_weight=0.5)
    # weekly decisions = the four Fridays
    fridays = [pd.Timestamp(d) for d in
               ("2020-01-10", "2020-01-17", "2020-01-24", "2020-01-31")]
    assert list(res.turnover.index) == fridays
    np.testing.assert_allclose(res.turnover.to_numpy(), [1.0, 0.0, 0.0, 0.0],
                               rtol=0, atol=TOL)
    assert abs(res.trading_costs.sum() - 2.0 * 0.001) < TOL  # initiation only
    assert res.daily_returns_gross.iloc[1:].abs().max() > 0  # weights did drift
    assert res.meta["n_skipped_unchanged_target"] == 3


def test_dollar_neutrality_and_cap_at_every_rebalance():
    rng = np.random.default_rng(11)
    n_days, n_names = 30, 12
    dates = list(pd.bdate_range("2020-01-06", periods=n_days))
    names = [f"T{i:02d}" for i in range(n_names)]
    close = {
        t: list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days))))
        for t in names
    }
    prices = _prices_long(dates, close)
    sig_rows = []
    for d_idx in (0, 10, 20):  # reshuffled scores -> targets change -> real trades
        perm = rng.permutation(n_names)
        sig_rows.append((dates[d_idx], {names[i]: float(perm[i]) for i in range(n_names)}))
    signal = _signal_long(sig_rows)
    res = run_ls_backtest(prices, signal, rebalance=10, execution="close",
                          n_quantiles=3, max_weight=0.3)
    assert res.meta["n_executed_trades"] == 3
    for d, grp in res.target_weights.groupby("date"):
        w = grp.set_index("ticker")["weight"]
        assert abs(w.sum()) < TOL                       # dollar neutral
        assert abs(w[w > 0].sum() - 1.0) < TOL          # long leg gross 1
        assert abs(w[w < 0].sum() + 1.0) < TOL          # short leg gross 1
        assert (w.abs() <= 0.3 + TOL).all()             # cap respected
    assert (res.turnover > 0).all()
    assert (res.turnover <= 2.0 + TOL).all()


# --------------------------------------------------------------------------- #
# costs in the backtest
# --------------------------------------------------------------------------- #

def test_cost_doubling_doubles_gross_net_gap():
    """Borrow-free single-rebalance case: doubled CostModel exactly doubles drag."""
    dates = list(pd.bdate_range("2020-01-06", periods=3))
    close = {"A": [100, 104, 108], "B": [100, 101, 99],
             "C": [100, 97, 96], "D": [100, 95, 93]}
    prices = _prices_long(dates, close)
    signal = _signal_long([(dates[0], {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})])
    kw = dict(rebalance=99, execution="close", n_quantiles=2, max_weight=0.5)
    res1 = run_ls_backtest(prices, signal,
                           costs=CostModel(half_spread_bps=25.0, commission_bps=5.0), **kw)
    res2 = run_ls_backtest(prices, signal,
                           costs=CostModel(half_spread_bps=25.0, commission_bps=5.0,
                                           doubled=True), **kw)
    pd.testing.assert_series_equal(res1.daily_returns_gross, res2.daily_returns_gross)
    gap1 = res1.daily_returns_gross - res1.daily_returns_net
    gap2 = res2.daily_returns_gross - res2.daily_returns_net
    np.testing.assert_allclose(gap2.to_numpy(), 2.0 * gap1.to_numpy(),
                               rtol=0, atol=1e-15)
    assert abs(gap1.iloc[0] - 2.0 * 0.003) < 1e-15      # initiation: traded 2.0 x 30bps
    assert (gap1.iloc[1:] == 0).all()


def test_borrow_accrues_daily_on_short_gross():
    dates = list(pd.bdate_range("2020-01-06", periods=4))
    close = {"X": [100.0] * 4, "Y": [100.0] * 4}
    prices = _prices_long(dates, close)
    signal = _signal_long([(dates[0], {"X": 2.0, "Y": 1.0})])
    cm = CostModel(borrow_bps_pa=252.0)                 # 1 bp per day
    res = run_ls_backtest(prices, signal, rebalance=1, execution="close",
                          n_quantiles=2, costs=cm, max_weight=1.0)
    assert (res.daily_returns_gross == 0).all()
    exp = [0.0] + [-cm.daily_borrow_rate] * 3           # short gross weight = 1.0
    np.testing.assert_allclose(res.daily_returns_net.to_numpy(), exp,
                               rtol=0, atol=1e-15)
    cm2 = CostModel(borrow_bps_pa=252.0, doubled=True)
    res2 = run_ls_backtest(prices, signal, rebalance=1, execution="close",
                           n_quantiles=2, costs=cm2, max_weight=1.0)
    np.testing.assert_allclose(res2.daily_returns_net.to_numpy(),
                               2.0 * res.daily_returns_net.to_numpy(),
                               rtol=0, atol=1e-15)


# --------------------------------------------------------------------------- #
# delistings, execution modes, calendars, skips
# --------------------------------------------------------------------------- #

def test_delisting_liquidation_at_last_close():
    n_days = 12
    dates = list(pd.bdate_range("2020-01-06", periods=n_days))
    close = {
        "n1": [100.0, 90.0, 81.0, 72.9, 65.61] + [None] * 7,  # dies after dates[4]
        "n2": [100.0] * n_days, "n3": [100.0] * n_days,
        "n4": [100.0] * n_days, "n5": [100.0] * n_days, "n6": [100.0] * n_days,
    }
    prices = _prices_long(dates, close)
    signal = _signal_long([(dates[0], {f"n{i}": float(i) for i in range(1, 7)})])
    res = run_ls_backtest(prices, signal, rebalance=7, execution="close",
                          n_quantiles=3, max_weight=1.0)
    # short n1 at -0.5 profits while it falls, then is closed at its last close
    assert res.n_delistings == 1
    assert res.delisted == ["n1"]
    assert res.meta["liquidations"] == [(dates[5], "n1")]
    assert abs(res.name_pnl["n1"] - 0.005 * (10.0 + 9.0 + 8.1 + 7.29)) < TOL
    g = res.daily_returns_gross
    assert np.isfinite(g.to_numpy()).all()
    assert abs(g.loc[dates[5]]) < TOL                   # cash after liquidation: no P&L
    assert abs(g.loc[dates[6]]) < TOL
    assert abs((1.0 + g).prod() - 1.17195) < TOL
    # next rebalance renormalizes over the 5 survivors (5 names, 3 quantiles)
    assert list(res.turnover.index) == [dates[0], dates[7]]
    assert res.turnover.loc[dates[7]] > 0.5
    tw7 = res.target_weights[res.target_weights["date"] == dates[7]]
    w7 = tw7.set_index("ticker")["weight"]
    assert abs(w7["n6"] - 1.0) < TOL
    assert abs(w7["n2"] + 0.5) < TOL
    assert abs(w7["n3"] + 0.5) < TOL
    assert abs(w7.sum()) < TOL


def test_next_open_execution():
    dates = list(pd.bdate_range("2020-01-06", periods=3))
    close = {"X": [100.0, 110.0, 121.0], "Y": [100.0] * 3}
    open_ = {"X": [100.0, 105.0, 116.0], "Y": [100.0] * 3}
    prices = _prices_long(dates, close, open_)
    signal = _signal_long([(dates[0], {"X": 2.0, "Y": 1.0})])

    res = run_ls_backtest(prices, signal, rebalance=99, execution="next_open",
                          n_quantiles=2, max_weight=1.0)
    # decided at d0, filled at open(d1)=105: day-1 return = (110-105)/105
    assert list(res.daily_returns_gross.index) == dates[1:]
    np.testing.assert_allclose(res.daily_returns_gross.to_numpy(),
                               [5.0 / 105.0, 0.1], rtol=0, atol=TOL)
    assert list(res.turnover.index) == [dates[1]]
    assert abs(res.turnover.iloc[0] - 1.0) < TOL
    np.testing.assert_allclose(res.long_returns.to_numpy(),
                               res.daily_returns_gross.to_numpy(), rtol=0, atol=TOL)
    np.testing.assert_allclose(res.short_returns.to_numpy(), [0.0, 0.0],
                               rtol=0, atol=TOL)       # Y flat: short contributes 0

    res_close = run_ls_backtest(prices, signal, rebalance=99, execution="close",
                                n_quantiles=2, max_weight=1.0)
    np.testing.assert_allclose(res_close.daily_returns_gross.to_numpy(),
                               [0.0, 0.1, 0.1], rtol=0, atol=TOL)

    with pytest.raises(ValueError):                    # next_open requires open column
        run_ls_backtest(_prices_long(dates, close), signal, rebalance=99,
                        execution="next_open", n_quantiles=2, max_weight=1.0)


def test_monthly_rebalance_calendar():
    dates = list(pd.bdate_range("2020-01-06", "2020-03-31"))
    close = {"A": [100.0 + k for k in range(len(dates))],
             "B": [100.0] * len(dates),
             "C": [100.0 - 0.2 * k for k in range(len(dates))],
             "D": [50.0 + 0.1 * k for k in range(len(dates))]}
    prices = _prices_long(dates, close)
    signal = _signal_long([(dates[0], {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})])
    res = run_ls_backtest(prices, signal, rebalance="M", execution="close",
                          n_quantiles=2, max_weight=0.5)
    month_ends = [pd.Timestamp(d) for d in ("2020-01-31", "2020-02-28", "2020-03-31")]
    assert list(res.turnover.index) == month_ends
    np.testing.assert_allclose(res.turnover.to_numpy(), [1.0, 0.0, 0.0],
                               rtol=0, atol=TOL)
    assert res.daily_returns_gross.index[0] == month_ends[0]


def test_small_cross_section_rebalances_skipped():
    dates = list(pd.bdate_range("2020-01-06", periods=5))
    close = {t: [100.0] * 5 for t in ("A", "B", "C", "D")}
    prices = _prices_long(dates, close)
    signal = _signal_long([
        (dates[0], {"A": 1.0, "B": 2.0, "C": 3.0}),              # only 3 names
        (dates[3], {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}),    # now 4
    ])
    res = run_ls_backtest(prices, signal, rebalance=1, execution="close",
                          n_quantiles=4, max_weight=1.0)
    assert res.skipped_rebalances == 3                 # d0..d2 too small
    assert res.meta["n_executed_trades"] == 1
    assert list(res.daily_returns_gross.index) == dates[3:]
    np.testing.assert_allclose(res.turnover.to_numpy(), [1.0, 0.0], rtol=0, atol=TOL)


def test_explicit_eligibility_blocks_stale_signal_after_exit():
    dates = list(pd.bdate_range("2020-01-06", periods=5))
    names = ["A", "B", "C", "D", "E", "F"]
    prices = _prices_long(dates, {t: [100.0] * len(dates) for t in names})
    signal = _signal_long([
        (dates[0], {t: float(i) for i, t in enumerate(names, start=1)}),
    ])
    eligibility = pd.DataFrame([
        {"ticker": t, "date": d, "eligible": not (t == "F" and d >= dates[2])}
        for d in dates
        for t in names
    ])

    res = run_ls_backtest(
        prices,
        signal,
        eligibility=eligibility,
        rebalance=2,
        execution="close",
        n_quantiles=3,
        max_weight=1.0,
    )

    first = res.target_weights[res.target_weights["date"] == dates[0]]
    second = res.target_weights[res.target_weights["date"] == dates[2]]
    assert "F" in set(first["ticker"])
    assert "F" not in set(second["ticker"])
    assert res.meta["explicit_eligibility"] is True


def test_eligibility_gap_resets_signal_carry_on_reentry():
    dates = list(pd.bdate_range("2020-01-06", periods=5))
    names = ["A", "B", "C", "D"]
    prices = _prices_long(dates, {t: [100.0] * len(dates) for t in names})
    signal = _signal_long([
        (dates[0], {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
        (dates[4], {"A": 4.0}),
    ])
    eligibility = pd.DataFrame([
        {"ticker": t, "date": d, "eligible": not (t == "A" and d in dates[1:3])}
        for d in dates
        for t in names
    ])

    res = run_ls_backtest(
        prices,
        signal,
        eligibility=eligibility,
        rebalance=1,
        execution="close",
        n_quantiles=2,
        max_weight=1.0,
    )

    # A is eligible again on d3, but its pre-exit score must not reappear until
    # the caller supplies a new score on d4.
    d3 = res.target_weights[res.target_weights["date"] == dates[3]]
    d4 = res.target_weights[res.target_weights["date"] == dates[4]]
    assert "A" not in set(d3["ticker"])
    assert "A" in set(d4["ticker"])


def test_eligibility_validation_fails_closed():
    dates = list(pd.bdate_range("2020-01-06", periods=2))
    prices = _prices_long(dates, {"A": [100.0, 100.0], "B": [100.0, 100.0]})
    signal = _signal_long([(dates[0], {"A": 2.0, "B": 1.0})])
    duplicate = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": [dates[0], dates[0]],
    })
    with pytest.raises(ValueError, match="duplicate ticker/date"):
        run_ls_backtest(
            prices,
            signal,
            eligibility=duplicate,
            rebalance=1,
            n_quantiles=2,
            max_weight=1.0,
        )


def test_missing_eligibility_liquidates_instead_of_holding_old_book():
    dates = list(pd.bdate_range("2020-01-06", periods=3))
    prices = _prices_long(dates, {
        "A": [100.0, 100.0, 100.0],
        "B": [100.0, 100.0, 100.0],
    })
    signal = _signal_long([(dates[0], {"A": 2.0, "B": 1.0})])
    eligibility = pd.DataFrame([
        {"ticker": ticker, "date": d, "eligible": d == dates[0]}
        for d in dates
        for ticker in ("A", "B")
    ])
    res = run_ls_backtest(
        prices,
        signal,
        eligibility=eligibility,
        rebalance=1,
        n_quantiles=2,
        max_weight=1.0,
    )
    assert res.meta["n_forced_cash_insufficient_eligibility"] == 1
    assert res.turnover.loc[dates[1]] == pytest.approx(1.0)
    assert res.turnover.loc[dates[2]] == 0.0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def test_summarize_promotion_gate_metrics():
    rng = np.random.default_rng(3)
    dates = list(pd.bdate_range("2020-01-01", "2021-12-31"))
    n_days = len(dates)
    close = {
        "A": [100.0 * 0.997 ** k for k in range(n_days)],
        "B": [100.0 * 0.998 ** k for k in range(n_days)],
        "C": list(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, n_days)))),
        "D": [100.0] * n_days,
        "E": [100.0 * 1.003 ** k for k in range(n_days)],
        "F": [100.0 * 1.004 ** k for k in range(n_days)],
    }
    prices = _prices_long(dates, close)
    signal = _signal_long([(dates[0], {t: float(i) for i, t in enumerate("ABCDEF")})])
    cm = CostModel(half_spread_bps=10.0, commission_bps=5.0, borrow_bps_pa=100.0)
    res = run_ls_backtest(prices, signal, rebalance="M", execution="close",
                          n_quantiles=3, costs=cm, max_weight=0.5)
    rep = summarize(res, by_year=True,
                    sector_map={"A": "fin", "B": "fin", "C": "mix",
                                "D": "mix", "E": "tech", "F": "tech"})

    for key in ("sharpe_gross_ann", "sharpe_net_ann", "max_drawdown_net",
                "turnover_oneway_ann", "breakeven_cost_bps", "beta_vs_ew_universe",
                "long_leg_total_return", "short_leg_total_return", "long_leg_share",
                "net_by_year", "max_year_share", "loyo_min_share",
                "top10_name_share", "max_sector_share", "flags", "notes"):
        assert key in rep, key
    assert rep["sharpe_gross_ann"] > 0
    assert rep["sharpe_net_ann"] < rep["sharpe_gross_ann"]
    assert rep["max_drawdown_net"] <= 0
    g = res.daily_returns_gross
    n = res.daily_returns_net
    years = len(n) / 252
    assert abs(rep["turnover_oneway_ann"] - res.turnover.sum() / years) < 1e-9
    assert abs(rep["breakeven_cost_bps"]
               - 1e4 * g.sum() / (2.0 * res.turnover.sum())) < 1e-9
    assert np.isfinite(rep["beta_vs_ew_universe"])
    assert abs(rep["long_leg_share"] + rep["short_leg_share"] - 1.0) < 1e-9
    assert set(rep["net_by_year"]) == {2020, 2021}
    shares = n.groupby(n.index.year).sum() / n.sum()
    assert abs(rep["max_year_share"] - shares.max()) < 1e-9
    assert abs(rep["loyo_min_share"] - (1.0 - shares.max())) < 1e-9
    assert rep["flags"]["one_year_gt_60pct"] == (shares.max() > 0.60)
    assert rep["top10_name_share"] == pytest.approx(1.0)   # only 6 names
    assert rep["flags"]["top10_names_gt_50pct"] is True
    assert abs(sum(rep["sector_pnl_share"].values()) - 1.0) < 1e-9
    assert isinstance(rep["flags"]["one_sector_gt_50pct"], bool)

    rep_nosec = summarize(res, by_year=True)
    assert rep_nosec["flags"]["one_sector_gt_50pct"] is None
    assert any("sector" in note for note in rep_nosec["notes"])


def test_summarize_single_year_concentration_and_breakeven():
    dates = list(pd.bdate_range("2020-01-06", periods=3))
    close = {"X": [100.0, 110.0, 121.0], "Y": [100.0] * 3}
    prices = _prices_long(dates, close)
    signal = _signal_long([(dates[0], {"X": 2.0, "Y": 1.0})])
    res = run_ls_backtest(prices, signal, rebalance=99, execution="close",
                          n_quantiles=2, max_weight=1.0)
    rep = summarize(res)
    assert abs(rep["max_year_share"] - 1.0) < TOL
    assert abs(rep["loyo_min_share"]) < TOL
    assert rep["flags"]["one_year_gt_60pct"] is True
    # single one-way turnover of 1.0: break-even/side = total gross P&L / 2 in bps
    assert abs(rep["breakeven_cost_bps"]
               - 1e4 * res.daily_returns_gross.sum() / 2.0) < 1e-9
    assert rep["flags"]["one_sector_gt_50pct"] is None


def test_offcalendar_signal_date_does_not_poison_eligibility():
    # Review F1: a signal stamped on a NON-TRADING day (weekend filing) must
    # inherit each name's as-of eligibility, not a synthetic all-False row
    # that resets every carry spell and force-liquidates the book.
    dates = list(pd.bdate_range("2020-01-06", periods=5))     # Mon..Fri
    saturday = dates[4] + pd.Timedelta(days=1)
    names = ["A", "B", "C", "D"]
    prices = _prices_long(dates + list(pd.bdate_range("2020-01-13", periods=3)),
                          {t: [100.0] * 8 for t in names})
    all_dates = list(pd.bdate_range("2020-01-06", periods=5)) + \
        list(pd.bdate_range("2020-01-13", periods=3))
    signal = _signal_long([
        (dates[0], {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
        (saturday, {"A": 1.0}),                               # weekend stamp
    ])
    eligibility = pd.DataFrame([
        {"ticker": t, "date": d, "eligible": True}
        for d in all_dates for t in names
    ])
    res = run_ls_backtest(prices, signal, eligibility=eligibility,
                          rebalance=1, execution="close",
                          n_quantiles=2, max_weight=1.0)
    # book survives the weekend stamp: no forced-cash event, positions on Monday
    assert res.meta["n_forced_cash_insufficient_eligibility"] == 0
    monday = res.target_weights[res.target_weights["date"] == all_dates[5]]
    assert len(monday) > 0
    # and the Saturday signal takes effect at the next decision: A drops to
    # the bottom quantile (its new score 1.0 < D's 1.0? equal ties are stable;
    # assert A is no longer alone in the top)
    top_monday = set(monday.loc[monday["weight"] > 0, "ticker"])
    assert "A" not in top_monday


def test_ongrid_absent_ticker_date_still_fails_closed():
    # The off-calendar guard must NOT weaken the documented contract: on a
    # date the eligibility frame DOES cover, an absent ticker-row = ineligible.
    dates = list(pd.bdate_range("2020-01-06", periods=4))
    names = ["A", "B", "C", "D"]
    prices = _prices_long(dates, {t: [100.0] * 4 for t in names})
    signal = _signal_long([(dates[0], {t: float(i) for i, t in
                                       enumerate(names, start=1)})])
    rows = [{"ticker": t, "date": d, "eligible": True}
            for d in dates for t in names
            if not (t == "D" and d == dates[2])]              # D's row absent on d2
    res = run_ls_backtest(prices, signal, eligibility=pd.DataFrame(rows),
                          rebalance=1, execution="close",
                          n_quantiles=2, max_weight=1.0)
    d2 = res.target_weights[res.target_weights["date"] == dates[2]]
    assert "D" not in set(d2["ticker"])                       # fail-closed held
    d3 = res.target_weights[res.target_weights["date"] == dates[3]]
    assert "D" not in set(d3["ticker"])                       # carry was reset too
