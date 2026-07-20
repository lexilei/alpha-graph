"""C32 earnings_streak builder: streak classification, chain breaks,
blanking rows, expiry rows."""
import numpy as np
import pandas as pd

from alpha_graph.signals.earnings_streak import build_earnings_streak


def sue_rows(ticker, quarters):
    """quarters: list of (period_end, sue) with disclosure ~35d later."""
    rows = []
    for pe, s in quarters:
        pe = pd.Timestamp(pe)
        rows.append(
            {"ticker": ticker, "period_end": pe,
             "disclosure_date": pe + pd.Timedelta(days=35), "sue": s}
        )
    return pd.DataFrame(rows)


def test_streak_and_nonstreak():
    q = [("2020-03-31", 1.0), ("2020-06-30", 2.0), ("2020-09-30", -1.0),
         ("2020-12-31", -0.5), ("2021-03-31", 3.0)]
    res = build_earnings_streak(sue_rows("A", q)).set_index("avail_date")
    vals = res["earnings_streak"]
    # q1: no prior -> NaN; q2: ++ streak -> 2.0; q3 sign flip -> NaN;
    # q4: -- streak -> -0.5; q5 flip -> NaN
    d = lambda pe: pd.Timestamp(pe) + pd.Timedelta(days=35)
    assert np.isnan(vals.loc[d("2020-03-31")])
    assert vals.loc[d("2020-06-30")] == 2.0
    assert np.isnan(vals.loc[d("2020-09-30")])
    assert vals.loc[d("2020-12-31")] == -0.5
    assert np.isnan(vals.loc[d("2021-03-31")])


def test_gap_breaks_chain():
    # a skipped quarter (>190d gap) makes the next announcement unclassifiable
    q = [("2020-03-31", 1.0), ("2020-12-31", 2.0), ("2021-03-31", 3.0)]
    res = build_earnings_streak(sue_rows("A", q)).set_index("avail_date")
    d = lambda pe: pd.Timestamp(pe) + pd.Timedelta(days=35)
    assert np.isnan(res["earnings_streak"].loc[d("2020-12-31")])  # 275d gap
    assert res["earnings_streak"].loc[d("2021-03-31")] == 3.0     # 90d, ++ streak


def test_expiry_rows_emitted():
    q = [("2020-03-31", 1.0), ("2020-06-30", 2.0)]
    res = build_earnings_streak(sue_rows("A", q))
    expiry_date = pd.Timestamp("2020-06-30") + pd.Timedelta(days=35 + 182)
    hit = res[res["avail_date"] == expiry_date]
    assert len(hit) == 1 and np.isnan(hit["earnings_streak"].iloc[0])
    # only value rows get expiry rows: 2 announcements + 1 expiry
    assert len(res) == 3
