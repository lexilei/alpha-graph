"""C33 revenue_surprise: tag-priority merge, share matching, drift-adjusted
surprise math on synthetic frames."""
import numpy as np
import pandas as pd

from alpha_graph.signals.revenue_surprise import (
    merge_by_priority,
    revenue_surprise_from_rps,
    rps_join,
)


def test_tag_priority_wins():
    def fact(tag, val, filed="2020-02-01"):
        return {"cik": "1", "ticker": "A", "tag": tag, "uom": "USD",
                "ddate": "2019-12-31", "qtrs": 1, "value": val,
                "adsh": f"a-{tag}", "filed": filed, "accepted": filed}
    facts = pd.DataFrame([
        fact("Revenues", 100.0),
        fact("RevenueFromContractWithCustomerExcludingAssessedTax", 999.0),
    ])
    merged, mix = merge_by_priority(facts)
    assert len(merged) == 1 and merged["value"].iloc[0] == 100.0
    assert mix == {"Revenues": 1.0}


def test_share_match_window():
    quarters = pd.DataFrame(
        [{"ticker": "A", "ddate": "2020-12-31", "eps": 1000.0,
          "stmt_filed": "2021-02-01"}]
    )
    near = pd.DataFrame(
        [{"ticker": "A", "end": "2021-01-15", "shares": 100.0}])
    far = pd.DataFrame(
        [{"ticker": "A", "end": "2021-06-30", "shares": 100.0}])
    assert rps_join(quarters, near)["rps"].iloc[0] == 10.0
    assert rps_join(quarters, far).empty  # outside +-60d
    brk = near.assign(ticker="BRK-B")
    assert rps_join(quarters.assign(ticker="BRK-B"), brk).empty


def test_drift_adjusted_surprise():
    # 13 quarters of RPS with constant seasonal growth 1.0 -> every delta
    # equals the drift, so RS should be 0 where defined... but a zero-std
    # prior window is excluded; add noise on the last quarter instead.
    dates = pd.date_range("2018-03-31", periods=13, freq="QE")
    vals = [10.0 + 0.25 * i for i in range(12)]  # delta = 1.0 each season
    vals.append(10.0 + 0.25 * 12 + 2.0)          # last quarter jumps
    rps = pd.DataFrame({"ticker": "A", "ddate": dates, "rps": vals,
                        "stmt_filed": dates + pd.Timedelta(days=35)})
    out = revenue_surprise_from_rps(rps)
    # constant-drift quarters have std=0 priors -> excluded; only the jump
    # quarter has dispersion... its own prior window is still constant.
    assert out.empty
    # add mild noise with period 3 (co-prime with the seasonal lag 4, so
    # deltas don't cancel to a constant) to give the priors spread
    vals2 = [10.0 + 0.25 * i + (0.05 if i % 3 == 0 else -0.05)
             for i in range(13)]
    vals2[12] += 2.0
    rps2 = rps.assign(rps=vals2)
    out2 = revenue_surprise_from_rps(rps2).set_index("ddate")
    last = out2["revenue_surprise"].loc[dates[12]]
    assert last > 3  # a 2.0 jump vs ~0.2-spread priors is a huge surprise
