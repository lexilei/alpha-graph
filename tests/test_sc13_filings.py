"""Synthetic tests for scripts/fetch_sc13_filings.py (loaded by path —
scripts/ is not a package; no network is touched at import).

Covers: form classification (only the four SC 13D/G types survive, SC 13E*
excluded) and canonicalization of EDGAR's post-2024-12-18 "SCHEDULE 13x"
vocabulary, the SUBJECT-role filter (blank Williams Act file number = the
corpus company is the FILER about someone else -> dropped), the 2010-07
window, key normalization (digits-only accession, zero-padded CIKs), the
amendment flag, dedup on accession (including across succession CIKs),
the OLD_CIK_MAP corpus augmentation (panel-ticker tagging, stale-map guards),
master.idx party parsing, filer resolution from idx parties (joint filings ->
lowest CIK; unresolved -> null), and the UTC -> ET acceptance conversion with
row-local mislabeled-ET rules A/B.
"""

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_sc13_filings.py"
_spec = importlib.util.spec_from_file_location("fetch_sc13_filings", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _raw_row(**over):
    row = dict(
        subject_cik="940944", subject_ticker="DRI",
        accession="0001521536-13-001064", form="SC 13D",
        filing_date="2013-12-23",
        acceptance_raw="2013-12-23T12:30:43.000Z", file_number="005-44577",
    )
    row.update(over)
    return row


# ------------------------------------------------------------------ normalize
def test_normalize_form_classification():
    raw = pd.DataFrame([
        _raw_row(),
        _raw_row(accession="0000000001-18-000001", form="SC 13G/A",
                 filing_date="2018-02-09"),
        _raw_row(accession="0000000001-18-000002", form="SC 13E4",
                 filing_date="2018-02-10"),          # going-private, not ownership
        _raw_row(accession="0000000001-18-000003", form="SC 13E3",
                 filing_date="2018-02-11"),
        _raw_row(accession="0000000001-18-000004", form=" SC 13G ",
                 filing_date="2018-02-12"),          # whitespace stripped
    ])
    out = mod.normalize(raw)
    assert sorted(out["form"]) == ["SC 13D", "SC 13G", "SC 13G/A"]


def test_normalize_canonicalizes_post_2024_schedule_vocabulary():
    # EDGAR renamed the form types on 2024-12-18 (structured 13D/G mandate);
    # filtering on the old names alone ends the panel at 2024-12-16.
    raw = pd.DataFrame([
        _raw_row(accession="0000000001-25-000001", form="SCHEDULE 13D",
                 filing_date="2025-01-06"),
        _raw_row(accession="0000000001-25-000002", form="SCHEDULE 13G/A",
                 filing_date="2025-02-14"),
    ])
    out = mod.normalize(raw)
    assert out["form"].tolist() == ["SC 13D", "SC 13G/A"]
    assert out["is_amendment"].tolist() == [False, True]


def test_normalize_subject_role_filter():
    raw = pd.DataFrame([
        _raw_row(),                                   # subject-role: 005 number
        _raw_row(accession="0000950157-19-000901", subject_ticker="OXY",
                 subject_cik="797468", filing_date="2019-08-19",
                 file_number=""),                     # OXY as FILER (about WES)
        _raw_row(accession="0000950103-23-011730", subject_ticker="CPB",
                 subject_cik="16732", filing_date="2023-08-08",
                 file_number=None),                   # missing == filer-role
    ])
    out = mod.normalize(raw)
    assert out["subject_ticker"].tolist() == ["DRI"]
    assert out.attrs["n_filer_role_dropped"] == 2


def test_normalize_window_keys_and_amendment_flag():
    raw = pd.DataFrame([
        _raw_row(accession="0000000001-10-000001", filing_date="2010-06-30"),  # pre-window
        _raw_row(accession="0000000001-10-000002", filing_date="2010-07-01"),  # boundary in
        _raw_row(accession="0000000001-15-000003", form="SC 13D/A",
                 filing_date="2015-03-02"),
    ])
    out = mod.normalize(raw)
    assert out["filing_date"].min() == pd.Timestamp("2010-07-01")
    assert out["accession"].tolist() == ["000000000110000002", "000000000115000003"]
    assert (out["subject_cik"] == "0000940944").all()
    assert out["is_amendment"].tolist() == [False, True]


def test_normalize_dedup_on_accession():
    raw = pd.DataFrame([_raw_row(), _raw_row(), _raw_row(subject_cik="940945")])
    out = mod.normalize(raw)
    assert len(out) == 1
    assert out.iloc[0]["subject_cik"] == "0000940944"  # lowest CIK wins, deterministic


def test_normalize_dedup_across_succession_ciks():
    # A filing listed under BOTH the old and the new CIK of the same company
    # (succession overlap) must survive exactly once, deterministically.
    raw = pd.DataFrame([
        _raw_row(subject_cik="1811074", subject_ticker="TPL",
                 accession="0000899140-19-000068", form="SC 13D",
                 filing_date="2019-03-15"),
        _raw_row(subject_cik="97517", subject_ticker="TPL",
                 accession="0000899140-19-000068", form="SC 13D",
                 filing_date="2019-03-15"),
    ])
    out = mod.normalize(raw)
    assert len(out) == 1
    assert out.iloc[0]["subject_cik"] == "0000097517"
    assert out.iloc[0]["subject_ticker"] == "TPL"


# ------------------------------------------------------------- old-CIK corpus
def test_old_cik_map_format():
    assert mod.OLD_CIK_MAP, "succession map must not be empty"
    all_ciks = [c for cs in mod.OLD_CIK_MAP.values() for c in cs]
    for t, ciks in mod.OLD_CIK_MAP.items():
        assert t == t.upper() and isinstance(ciks, list) and ciks
        for c in ciks:
            assert len(c) == 10 and c.isdigit()  # zero-padded, joins subject_cik
    assert len(all_ciks) == len(set(all_ciks))


def test_augment_corpus_tags_old_ciks_with_panel_ticker(monkeypatch):
    monkeypatch.setattr(mod, "OLD_CIK_MAP", {"TPL": ["0000097517"]})
    corpus = pd.DataFrame([{"cik": "0001811074", "ticker": "TPL"},
                           {"cik": "0000320193", "ticker": "AAPL"}])
    out = mod.augment_corpus_with_old_ciks(corpus)
    assert len(out) == 3
    assert out.loc[out["cik"] == "0000097517", "ticker"].tolist() == ["TPL"]
    assert set(corpus["cik"]) < set(out["cik"])  # current corpus untouched


def test_augment_corpus_rejects_stale_map(monkeypatch):
    corpus = pd.DataFrame([{"cik": "0001811074", "ticker": "TPL"}])
    # old CIK colliding with a live corpus CIK
    monkeypatch.setattr(mod, "OLD_CIK_MAP", {"TPL": ["0001811074"]})
    with pytest.raises(ValueError, match="collides"):
        mod.augment_corpus_with_old_ciks(corpus)
    # map ticker that left the panel
    monkeypatch.setattr(mod, "OLD_CIK_MAP", {"ZZZ": ["0000097517"]})
    with pytest.raises(ValueError, match="not in the corpus"):
        mod.augment_corpus_with_old_ciks(corpus)


# ------------------------------------------------------------- master.idx
def test_parse_master_idx():
    text = (
        "Description: Master Index of EDGAR Dissemination Feed\n"
        "CIK|Company Name|Form Type|Date Filed|Filename\n"
        "--------------------------------------------------------------\n"
        "1517137|Starboard Value LP|SC 13D|2013-12-23|"
        "edgar/data/1517137/0001521536-13-001064.txt\n"
        "940944|DARDEN RESTAURANTS INC|SC 13D|2013-12-23|"
        "edgar/data/940944/0001521536-13-001064.txt\n"
        "940944|DARDEN RESTAURANTS INC|8-K|2013-12-20|edgar/data/940944/x.txt\n"
        "not|a|valid\n"
    )
    rows = mod.parse_master_idx(text, "2013Q4")
    assert len(rows) == 2  # 8-K and malformed lines skipped
    assert rows[0] == {
        "quarter": "2013Q4", "cik": "0001517137", "name": "Starboard Value LP",
        "form": "SC 13D", "date": "2013-12-23", "accession": "000152153613001064",
    }


def test_parse_master_idx_canonicalizes_new_vocabulary():
    text = ("102909|VANGUARD GROUP INC|SCHEDULE 13G/A|2025-02-11|"
            "edgar/data/102909/0001102909-25-000123.txt\n")
    rows = mod.parse_master_idx(text, "2025Q1")
    assert rows == [{
        "quarter": "2025Q1", "cik": "0000102909", "name": "VANGUARD GROUP INC",
        "form": "SC 13G/A", "date": "2025-02-11", "accession": "000110290925000123",
    }]


def test_quarters_through():
    qs = mod.quarters_through(date(2011, 1, 5))
    assert qs == ["2010Q3", "2010Q4", "2011Q1"]


# ------------------------------------------------------------- filer resolution
def _spine(rows):
    return pd.DataFrame(rows, columns=["accession", "subject_cik", "subject_ticker"])


def test_attach_filers_resolves_non_subject_party():
    spine = _spine([("000152153613001064", "0000940944", "DRI")])
    idx = pd.DataFrame([
        dict(accession="000152153613001064", cik="0001517137", name="Starboard Value LP"),
        dict(accession="000152153613001064", cik="0000940944", name="DARDEN RESTAURANTS INC"),
    ])
    out = mod.attach_filers(spine, idx)
    assert out.loc[0, "filer_cik"] == "0001517137"
    assert out.loc[0, "filer_name"] == "Starboard Value LP"
    assert out.loc[0, "n_filers"] == 1


def test_attach_filers_joint_filing_lowest_cik():
    spine = _spine([("000000000120000001", "0000797468", "OXY")])
    idx = pd.DataFrame([
        dict(accession="000000000120000001", cik="0001173073", name="High River LP"),
        dict(accession="000000000120000001", cik="0000921669", name="Icahn Carl C"),
        dict(accession="000000000120000001", cik="0000797468", name="OCCIDENTAL PETROLEUM"),
    ])
    out = mod.attach_filers(spine, idx)
    assert out.loc[0, "filer_cik"] == "0000921669"  # lowest CIK, deterministic
    assert out.loc[0, "filer_name"] == "Icahn Carl C"
    assert out.loc[0, "n_filers"] == 2


def test_attach_filers_unresolved_is_null():
    spine = _spine([("000000000120000001", "0000940944", "DRI"),   # not in idx
                    ("000000000120000002", "0000940944", "DRI")])  # subject-only in idx
    idx = pd.DataFrame([
        dict(accession="000000000120000002", cik="0000940944", name="DARDEN RESTAURANTS INC"),
    ])
    out = mod.attach_filers(spine, idx)
    assert out["filer_cik"].isna().all()
    assert out["filer_name"].isna().all()
    assert out["n_filers"].tolist() == [0, 0]


# ------------------------------------------------------------- acceptance ts
def test_to_eastern_plain_conversion():
    raw = pd.Series(["2013-12-23T12:30:43.000Z",   # EST: UTC-5
                     "2019-08-19T20:18:41.000Z",   # EDT: UTC-4
                     None])
    fdate = pd.to_datetime(pd.Series(["2013-12-23", "2019-08-19", "2020-01-01"]))
    ts, fixed = mod.to_eastern(raw, fdate)
    assert ts[0] == pd.Timestamp("2013-12-23 07:30:43")
    assert ts[1] == pd.Timestamp("2019-08-19 16:18:41")
    assert pd.isna(ts[2])
    assert not fixed.any()


def test_to_eastern_rule_a_out_of_window():
    # API stores ET wall time with a spurious Z: 06:30 "UTC" -> 01:30 ET is
    # outside EDGAR's 06:00-22:05 acceptance window; raw wall time is inside.
    raw = pd.Series(["2020-01-15T06:30:10.000Z"])
    ts, fixed = mod.to_eastern(raw, pd.to_datetime(pd.Series(["2020-01-15"])))
    assert bool(fixed[0])
    assert ts[0] == pd.Timestamp("2020-01-15 06:30:10")


def test_to_eastern_rule_b_dissemination_inconsistent():
    # UTC-parsed 12:45 ET is before the 17:30 cutoff yet the filing date is the
    # NEXT day (impossible); raw wall time 17:45 >= 17:20 is consistent.
    raw = pd.Series(["2020-01-15T17:45:00.000Z"])
    ts, fixed = mod.to_eastern(raw, pd.to_datetime(pd.Series(["2020-01-16"])))
    assert bool(fixed[0])
    assert ts[0] == pd.Timestamp("2020-01-15 17:45:00")


def test_output_schema():
    assert mod.COLUMNS == [
        "subject_ticker", "subject_cik", "accession", "form", "is_amendment",
        "filing_date", "acceptance_ts", "filer_name", "filer_cik",
    ]
