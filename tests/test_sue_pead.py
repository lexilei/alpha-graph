"""Tests for C17 `sue_pead` (synthetic inputs only — no real caches touched).

Pinned semantics under test (registration 2026-07-13 + the build-time
std-exclusion pin):
  - first-filed discipline beats later revisions of the same (cik,ddate,qtrs);
  - Q4 derivation is exact, skipped when a fiscal quarter is missing, and
    inherits the ANNUAL filing's availability;
  - the seasonal match is 365±45 days (a two-year gap is NOT bridged);
  - SUE = diff / std(trailing 8 diffs EXCLUDING the current one, min 6,
    ddof=1), std==0 -> NaN;
  - availability = earliest Item-2.02 8-K date in (period_end, filed], else
    the statement filed date; acceptance at/after 16:00 ET shifts it +1 day;
  - PIT: appending later filings changes nothing dated before them.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_graph.signals.sue_pead import EPS_TAG, compute, first_filed_eps

EMPTY_8K = pd.DataFrame({"ticker": pd.Series(dtype=str),
                         "fdate": pd.Series(dtype="datetime64[ns]"),
                         "accession": pd.Series(dtype=str),
                         "has_202": pd.Series(dtype=bool)})
EMPTY_ACC = pd.DataFrame({"accession": pd.Series(dtype=str),
                          "acceptance_ts": pd.Series(dtype="datetime64[ns]")})


def _fact(cik="0000000001", ticker="AAA", ddate="2020-03-31", qtrs=1, value=1.0,
          tag=EPS_TAG, uom="USD", filed="2020-05-01", accepted=None, adsh="a1",
          period=None, form="10-Q"):
    filed = pd.Timestamp(filed)
    return dict(
        cik=cik, ticker=ticker, ddate=pd.Timestamp(ddate), qtrs=qtrs,
        value=value, tag=tag, uom=uom, filed=filed,
        accepted=pd.Timestamp(accepted) if accepted else filed + pd.Timedelta(hours=12),
        adsh=adsh, period=pd.Timestamp(period) if period else pd.Timestamp(ddate),
        form=form,
    )


def _compute(facts_rows, eightk=None, acceptance=None):
    return compute(
        facts=pd.DataFrame(facts_rows),
        eightk=EMPTY_8K if eightk is None else eightk,
        acceptance=EMPTY_ACC if acceptance is None else acceptance,
    )


# --------------------------------------------------------------------------- #
# First-filed discipline
# --------------------------------------------------------------------------- #

def test_first_filed_beats_revision():
    # Same (cik, ddate, qtrs) in two accessions: the original 10-Q and next
    # year's comparative restating it. The original value must win.
    facts = pd.DataFrame([
        _fact(value=1.00, filed="2020-05-01", adsh="orig"),
        _fact(value=9.99, filed="2021-05-01", adsh="restated", period="2021-03-31"),
    ])
    out = first_filed_eps(facts)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 1.00
    assert out.iloc[0]["adsh"] == "orig"


def test_non_per_share_uom_rows_dropped():
    facts = pd.DataFrame([
        _fact(value=1.00, adsh="good"),
        _fact(value=123.0, adsh="junk", uom="shares", filed="2020-04-01"),
    ])
    out = first_filed_eps(facts)
    assert len(out) == 1
    assert out.iloc[0]["adsh"] == "good"      # earlier-filed junk uom ignored


# --------------------------------------------------------------------------- #
# Q4 derivation
# --------------------------------------------------------------------------- #

def _fy2020(include=("Q1", "Q2", "Q3"), annual=4.0, q4_standalone=None):
    qd = {"Q1": ("2020-03-31", 1.0, "2020-05-01"),
          "Q2": ("2020-06-30", 0.5, "2020-08-01"),
          "Q3": ("2020-09-30", 1.5, "2020-11-01")}
    rows = [_fact(ddate=d, value=v, filed=f, adsh=f"q{k}")
            for k, (d, v, f) in qd.items() if k in include]
    if annual is not None:
        rows.append(_fact(ddate="2020-12-31", qtrs=4, value=annual,
                          filed="2021-02-15", adsh="ann10k",
                          period="2020-12-31", form="10-K"))
    if q4_standalone is not None:
        rows.append(_fact(ddate="2020-12-31", qtrs=1, value=q4_standalone,
                          filed="2021-02-15", adsh="ann10k",
                          period="2020-12-31", form="10-K"))
    return rows


def test_q4_derivation_exact_and_annual_availability():
    df = _compute(_fy2020())
    q4 = df[df["period_end"] == pd.Timestamp("2020-12-31")]
    assert len(q4) == 1
    r = q4.iloc[0]
    assert bool(r["derived_q4"])
    assert r["eps_q"] == 1.0                                   # 4.0 - (1+0.5+1.5), exact
    # availability = the ANNUAL filing's availability (fallback: its filed date)
    assert r["disclosure_date"] == pd.Timestamp("2021-02-15")
    assert not bool(r["via_8k02"])


def test_q4_derivation_skipped_when_quarter_missing():
    df = _compute(_fy2020(include=("Q1", "Q3")))
    assert (df["period_end"] != pd.Timestamp("2020-12-31")).all()


def test_q4_standalone_wins_over_derivation():
    df = _compute(_fy2020(q4_standalone=0.7))
    q4 = df[df["period_end"] == pd.Timestamp("2020-12-31")]
    assert len(q4) == 1
    assert q4.iloc[0]["eps_q"] == 0.7
    assert not bool(q4.iloc[0]["derived_q4"])


# --------------------------------------------------------------------------- #
# Seasonal match tolerance
# --------------------------------------------------------------------------- #

def test_seasonal_match_skips_gap_year():
    # 2020-03-31 quarter missing: 2021-03-31 must NOT match 2019-03-31 (730d)
    df = _compute([
        _fact(ddate="2019-03-31", value=1.0, filed="2019-05-01", adsh="a"),
        _fact(ddate="2021-03-31", value=2.0, filed="2021-05-01", adsh="b"),
    ])
    r = df[df["period_end"] == pd.Timestamp("2021-03-31")].iloc[0]
    assert np.isnan(r["eps_q4_ago"])


def test_seasonal_match_leap_year_and_tolerance_boundary():
    # 366-day gap (leap year) matches; 410 days (=365+45) matches; 411 does not.
    df = _compute(
        [_fact(cik="1", ticker="A", ddate="2019-06-30", value=1.0,
               filed="2019-08-01", adsh="a1"),
         _fact(cik="1", ticker="A", ddate="2020-06-30", value=1.5,
               filed="2020-08-01", adsh="a2"),
         _fact(cik="2", ticker="B", ddate="2019-01-01", value=2.0,
               filed="2019-02-15", adsh="b1"),
         _fact(cik="2", ticker="B", ddate="2020-02-15", value=2.5,   # +410d
               filed="2020-03-20", adsh="b2"),
         _fact(cik="3", ticker="C", ddate="2019-01-01", value=3.0,
               filed="2019-02-15", adsh="c1"),
         _fact(cik="3", ticker="C", ddate="2020-02-16", value=3.5,   # +411d
               filed="2020-03-20", adsh="c2")])
    by = df.set_index(["ticker", "period_end"])["eps_q4_ago"]
    assert by[("A", pd.Timestamp("2020-06-30"))] == 1.0
    assert by[("B", pd.Timestamp("2020-02-15"))] == 2.0
    assert np.isnan(by[("C", pd.Timestamp("2020-02-16"))])


# --------------------------------------------------------------------------- #
# SUE arithmetic
# --------------------------------------------------------------------------- #

QENDS = pd.date_range("2015-03-31", periods=16, freq="QE")
VALS = [1.0, 1.1, 0.9, 1.2, 1.3, 1.0, 1.4, 1.1,
        1.6, 1.2, 1.3, 1.5, 100.0, 1.4, 1.7, 1.9]


def _series_facts(vals=VALS, qends=QENDS):
    return [_fact(ddate=d, value=v, filed=d + pd.Timedelta(days=35), adsh=f"a{i:02d}")
            for i, (d, v) in enumerate(zip(qends, vals))]


def test_sue_hand_computed_and_std_excludes_current():
    df = _compute(_series_facts()).set_index("period_end")
    eps = np.asarray(VALS)
    diffs = eps[4:] - eps[:-4]                     # diff of quarter i is diffs[i-4]
    # Quarter 12 carries a huge shock (100.0). Its null must be the 8 PRIOR
    # diffs (quarters 4..11) only — the current diff must not sit in its own
    # denominator.
    expect = diffs[8] / np.std(diffs[0:8], ddof=1)
    got = df.loc[QENDS[12], "sue"]
    assert got == pytest.approx(expect, abs=1e-12)
    # An inclusive window (quarters 5..12) would give a very different value.
    inclusive = diffs[8] / np.std(diffs[1:9], ddof=1)
    assert abs(got - inclusive) > 1.0


def test_sue_min6_gate():
    df = _compute(_series_facts()).set_index("period_end")
    eps = np.asarray(VALS)
    diffs = eps[4:] - eps[:-4]
    # First diff exists at quarter 4; quarter 9 has only 5 prior diffs -> NaN.
    assert np.isnan(df.loc[QENDS[9], "sue"])
    # Quarter 10 has exactly 6 prior diffs -> defined, hand-checked.
    expect = diffs[6] / np.std(diffs[0:6], ddof=1)
    assert df.loc[QENDS[10], "sue"] == pytest.approx(expect, abs=1e-12)


def test_sue_zero_std_is_nan():
    # eps grows by exactly +0.5 YoY -> every seasonal diff is 0.5 -> std 0.
    vals = [1.0, 2.0, 3.0, 4.0]
    for i in range(12):
        vals.append(vals[i] + 0.5)
    df = _compute(_series_facts(vals=vals)).set_index("period_end")
    assert np.isnan(df.loc[QENDS[12], "sue"])


# --------------------------------------------------------------------------- #
# Availability: Item-2.02 8-K vs statement filed date
# --------------------------------------------------------------------------- #

def _one_quarter():
    return [_fact(ddate="2020-03-31", value=1.0, filed="2020-05-05", adsh="q1")]


def _acc(accession, ts):
    return {"accession": accession, "acceptance_ts": pd.Timestamp(ts)}


def _e8k(ticker, fdate, accession, has_202):
    return {"ticker": ticker, "fdate": pd.Timestamp(fdate),
            "accession": accession, "has_202": has_202}


def test_availability_prefers_earliest_item_202_8k():
    eightk = pd.DataFrame([
        _e8k("AAA", "2020-03-31", "e0", True),    # ON period_end: excluded (open bound)
        _e8k("AAA", "2020-04-10", "e1", False),   # in window but not 2.02
        _e8k("AAA", "2020-04-20", "e2", True),    # chosen
        _e8k("AAA", "2020-04-25", "e3", True),    # later 2.02
    ])
    acc = pd.DataFrame([_acc("e2", "2020-04-20 08:30:00")])
    r = compute(facts=pd.DataFrame(_one_quarter()), eightk=eightk,
                acceptance=acc).iloc[0]
    assert r["disclosure_date"] == pd.Timestamp("2020-04-20")
    assert bool(r["via_8k02"])


def test_availability_falls_back_to_statement_filed():
    eightk = pd.DataFrame([
        _e8k("AAA", "2020-04-10", "e1", False),   # non-2.02 only
        _e8k("AAA", "2020-05-06", "e4", True),    # 2.02 but AFTER filed date
    ])
    r = compute(facts=pd.DataFrame(_one_quarter()), eightk=eightk,
                acceptance=EMPTY_ACC).iloc[0]
    assert r["disclosure_date"] == pd.Timestamp("2020-05-05")   # filed date
    assert not bool(r["via_8k02"])


def test_acceptance_at_or_after_1600_shifts_one_day():
    eightk = pd.DataFrame([_e8k("AAA", "2020-04-20", "e2", True)])
    acc = pd.DataFrame([_acc("e2", "2020-04-20 16:00:00")])
    r = compute(facts=pd.DataFrame(_one_quarter()), eightk=eightk,
                acceptance=acc).iloc[0]
    assert r["disclosure_date"] == pd.Timestamp("2020-04-21")
    assert bool(r["via_8k02"])


def test_acceptance_before_1600_no_shift():
    eightk = pd.DataFrame([_e8k("AAA", "2020-04-20", "e2", True)])
    acc = pd.DataFrame([_acc("e2", "2020-04-20 15:59:59")])
    r = compute(facts=pd.DataFrame(_one_quarter()), eightk=eightk,
                acceptance=acc).iloc[0]
    assert r["disclosure_date"] == pd.Timestamp("2020-04-20")


# --------------------------------------------------------------------------- #
# PIT perturbation
# --------------------------------------------------------------------------- #

def test_pit_later_filing_changes_nothing_before_it():
    base_facts = _series_facts()
    base = _compute(base_facts)

    # Append strictly later material: a revision of an old quarter, a brand-new
    # quarter, and a late Item-2.02 8-K. Nothing dated before them may move.
    pert_facts = base_facts + [
        _fact(ddate="2015-03-31", value=55.0, filed="2026-01-05", adsh="zrev",
              period="2025-12-31"),
        _fact(ddate=QENDS[-1] + pd.Timedelta(days=91), value=2.0,
              filed="2026-02-01", adsh="znew"),
    ]
    eightk = pd.DataFrame([_e8k("AAA", "2026-01-20", "z8k", True)])
    acc = pd.DataFrame([_acc("z8k", "2026-01-20 09:00:00")])
    pert = compute(facts=pd.DataFrame(pert_facts), eightk=eightk, acceptance=acc)

    old = pert[pert["period_end"].isin(base["period_end"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, base)
