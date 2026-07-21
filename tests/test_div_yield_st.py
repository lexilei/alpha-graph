"""C36 div_yield_st: PIT frequency inference, lag arithmetic, payer
universe, zero semantics."""
import numpy as np
import pandas as pd

from alpha_graph.signals.div_yield_st import build_div_yield_st


def market_frame(ticker, months, close=100.0):
    rows = []
    for m in months:
        p = pd.Period(m, freq="M")
        days = pd.bdate_range(p.start_time, p.end_time)
        rows += [{"ticker": ticker, "date": d, "close": close} for d in days]
    return pd.DataFrame(rows)


def quarterly_divs(ticker, first="2018-02-10", n=20, amount=1.0):
    dates = [pd.Timestamp(first) + pd.DateOffset(months=3 * i) for i in range(n)]
    return pd.DataFrame(
        {"ticker": ticker, "ex_date": dates, "amount": amount}
    )


def test_quarterly_prediction_and_zeros():
    months = [f"2021-{mm:02d}" for mm in range(1, 13)]
    div = quarterly_divs("A")  # ex-months: Feb, May, Aug, Nov
    res = build_div_yield_st(div, market_frame("A", months))
    res["m"] = res["date"].dt.to_period("M")
    by = res.set_index("m")["div_yield_st"]
    # signal at month m = amount if ex-div at m-2: Apr/Jul/Oct/Jan predict
    # next-month (May/Aug/Nov/Feb) payments -> yield 1/100
    assert by[pd.Period("2021-04")] == 0.01
    assert by[pd.Period("2021-07")] == 0.01
    # months without an ex-div 2 months prior are explicit zeros
    assert by[pd.Period("2021-05")] == 0.0
    # every month is in the payer universe (paid within 12m)
    assert len(by) == 12


def test_nonpayer_excluded_and_pit_frequency():
    months = [f"2021-{mm:02d}" for mm in range(1, 7)]
    # payer B stopped paying in 2018: outside trailing 12m -> excluded
    old = quarterly_divs("B", first="2017-02-10", n=6)
    res = build_div_yield_st(old, market_frame("B", months))
    assert res.empty
    # ticker C has only 2 ex-dates in the trailing 36m -> unclassifiable
    sparse = pd.DataFrame(
        {"ticker": "C",
         "ex_date": [pd.Timestamp("2020-06-10"), pd.Timestamp("2021-01-05")],
         "amount": 1.0}
    )
    res_c = build_div_yield_st(sparse, market_frame("C", months))
    assert res_c.empty


def test_annual_payer_lag_11():
    months = [f"2021-{mm:02d}" for mm in range(1, 13)]
    dates = [pd.Timestamp(f"{y}-03-15") for y in (2018, 2019, 2020, 2021)]
    div = pd.DataFrame({"ticker": "D", "ex_date": dates, "amount": 2.0})
    res = build_div_yield_st(div, market_frame("D", months))
    res["m"] = res["date"].dt.to_period("M")
    by = res.set_index("m")["div_yield_st"]
    # annual: signal at 2022-02 would predict Mar; within 2021, signal at
    # Feb 2021 = amount from Mar 2020 (11 months prior) -> 2/100
    assert by[pd.Period("2021-02")] == 0.02
    assert (by.drop(pd.Period("2021-02")) == 0.0).all()
