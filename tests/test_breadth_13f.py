"""C35 breadth_13f: first-filed selection, deadline cut, filters,
partial-period guard, quarter-adjacent delta."""
import numpy as np
import pandas as pd

from alpha_graph.signals.breadth_13f import build_breadth


def rows(acc, cik, period, filed, tickers, subtype="13F-HR",
         shtype="SH", putcall=None, shares="100"):
    return [
        {"ACCESSION_NUMBER": acc, "CIK": cik, "FILING_DATE": filed,
         "SUBMISSIONTYPE": subtype, "PERIODOFREPORT": period,
         "ticker": t, "SSHPRNAMT": shares, "SSHPRNAMTTYPE": shtype,
         "PUTCALL": putcall}
        for t in tickers
    ]


def write_files(tmp_path, frames):
    paths = []
    for i, fr in enumerate(frames):
        p = tmp_path / f"f{i}.parquet"
        pd.DataFrame(fr).to_parquet(p, index=False)
        paths.append(str(p))
    return paths


def test_breadth_delta_and_filters(tmp_path):
    # Build 6 quarters so the rolling guard has stable neighbors; two
    # filers in q5 for AAPL (breadth 1 -> 2), with excluded rows around.
    frames = []
    periods = ["31-MAR-2020", "30-JUN-2020", "30-SEP-2020", "31-DEC-2020",
               "31-MAR-2021", "30-JUN-2021"]
    fileds = ["30-APR-2020", "30-JUL-2020", "30-OCT-2020", "30-JAN-2021",
              "30-APR-2021", "30-JUL-2021"]
    for i, (pe, fd) in enumerate(zip(periods, fileds)):
        f = []
        f += rows(f"a{i}", "1111", pe, fd, ["AAPL", "MSFT"])
        if i >= 4:  # second holder joins AAPL in 2021
            f += rows(f"b{i}", "2222", pe, fd, ["AAPL"])
        # excluded flavors: amendment, put option, PRN, late filing
        f += rows(f"x{i}", "3333", pe, fd, ["AAPL"], subtype="13F-HR/A")
        f += rows(f"y{i}", "4444", pe, fd, ["AAPL"], putcall="Put")
        f += rows(f"z{i}", "5555", pe, fd, ["AAPL"], shtype="PRN")
        f += rows(f"w{i}", "6666", pe, "30-DEC-2021", ["AAPL"])  # late
        frames.append(f)
    res = build_breadth(write_files(tmp_path, frames)).set_index(
        ["ticker", "avail_date"]
    )
    d = pd.Timestamp("2021-03-31") + pd.Timedelta(days=46)
    # AAPL breadth 1 -> 2 at 2021q1: delta +1; MSFT stays 1: delta 0
    assert res.loc[("AAPL", d), "breadth_13f"] == 1.0
    assert res.loc[("MSFT", d), "breadth_13f"] == 0.0
    # deltas exist only for quarter-adjacent pairs; 5 deltas per ticker
    assert len(res.loc["AAPL"]) == 5


def test_first_filed_wins(tmp_path):
    # same cik files twice for one period: earliest accession counts
    f = rows("early", "1111", "31-MAR-2020", "15-APR-2020", ["AAPL"])
    f += rows("late", "1111", "31-MAR-2020", "30-APR-2020", ["AAPL", "MSFT"])
    f += rows("q2", "1111", "30-JUN-2020", "30-JUL-2020", ["AAPL", "MSFT"])
    res = build_breadth(write_files(tmp_path, [f]))
    # q1 breadth: AAPL=1 (early accession), MSFT=0 -> q2 delta MSFT undefined
    # (no prior breadth row exists for MSFT), AAPL delta = 0
    aapl = res[res["ticker"] == "AAPL"]
    assert len(aapl) == 1 and aapl["breadth_13f"].iloc[0] == 0.0
    assert (res["ticker"] == "MSFT").sum() == 0
