"""Tests for the PIT membership module (plan v2, Item 1).

The real-CSV tests are the plan's acceptance gates: TSLA exclusion (a late
joiner has no pre-join membership) and the META-2016 false-drop gate (a
renamed continuous member keeps its membership through the rename map).
"""

import pandas as pd
import pytest

from alpha_graph.data.pit_universe import (
    MEMBERSHIP_CSV,
    load_membership,
    membership_asof,
    membership_mask,
)

needs_csv = pytest.mark.skipif(
    not MEMBERSHIP_CSV.exists(), reason="membership CSV not present")


@pytest.fixture(scope="module")
def snaps():
    if not MEMBERSHIP_CSV.exists():
        pytest.skip("membership CSV not present")
    return load_membership()


# --------------------------------------------------------------------------- #
# Acceptance gates on the real file
# --------------------------------------------------------------------------- #

@needs_csv
def test_tsla_excluded_before_join(snaps):
    # TSLA joined the S&P 500 on 2020-12-21 (effective-dated in the CSV)
    assert "TSLA" not in membership_asof(snaps, pd.Timestamp("2020-11-30"))
    assert "TSLA" in membership_asof(snaps, pd.Timestamp("2020-12-31"))


@needs_csv
def test_meta_false_drop_gate(snaps):
    # META was a member throughout 2016 under its old symbol FB; the rename
    # map must carry that membership. This test fails without the map.
    assert "META" in membership_asof(snaps, pd.Timestamp("2016-06-30"))
    assert "FB" not in membership_asof(snaps, pd.Timestamp("2016-06-30"))


@needs_csv
def test_rename_chains(snaps):
    d = pd.Timestamp("2016-06-30")
    s = membership_asof(snaps, d)
    # flattened chains: Symantec-era GEN, HCP-era DOC, CBS-era PSKY,
    # MeadWestvaco/WestRock-era SW, UTX-era RTX
    for t in ("GEN", "DOC", "PSKY", "SW", "RTX", "ELV", "WTW"):
        assert t in s, t


@needs_csv
def test_ir_dated_reuse_rule(snaps):
    # Through Feb-2020 the CSV's "IR" is Ingersoln-Rand plc = today's TT;
    # panel IR (ex-Gardner-Denver) must NOT inherit that membership.
    d = pd.Timestamp("2019-06-28")
    s = membership_asof(snaps, d)
    assert "TT" in s and "IR" not in s
    # after the reuse date the new IR is the member and TT stands alone
    d2 = pd.Timestamp("2021-06-30")
    s2 = membership_asof(snaps, d2)
    assert "IR" in s2 and "TT" in s2


@needs_csv
def test_carry_forward_bound(snaps):
    last = snaps[-1][0]
    assert membership_asof(snaps, last + pd.Timedelta(days=30)) is not None
    assert membership_asof(snaps, last + pd.Timedelta(days=90)) is None
    assert membership_asof(snaps, snaps[0][0] - pd.Timedelta(days=1)) is None


# --------------------------------------------------------------------------- #
# Synthetic: gap handling and the mask
# --------------------------------------------------------------------------- #

def _synth():
    # three snapshots with a 100-day gap; XS_BAND is bypassed via rename_map
    # trick? No — build sets inside the band by padding.
    pad = [f"P{i:03d}" for i in range(494)]
    s1 = frozenset(pad + ["AAA", "BBB"])
    s2 = frozenset(pad + ["AAA", "CCC"])          # BBB out, CCC in
    return [
        (pd.Timestamp("2020-01-01"), s1),
        (pd.Timestamp("2020-01-03"), s1),
        (pd.Timestamp("2020-04-12"), s2),          # 100-day gap: legal
    ]


def test_asof_fill_over_long_gap():
    snaps = _synth()
    # inside the gap: last snapshot before the change still applies
    s = membership_asof(snaps, pd.Timestamp("2020-03-01"))
    assert "BBB" in s and "CCC" not in s
    s2 = membership_asof(snaps, pd.Timestamp("2020-04-12"))
    assert "CCC" in s2 and "BBB" not in s2


def test_membership_mask_rows():
    snaps = _synth()
    panel = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC", "CCC"],
        "date": pd.to_datetime(["2020-02-01", "2020-05-01",
                                "2020-02-01", "2020-05-01"]),
    })
    m = membership_mask(panel, snaps)
    assert list(m) == [True, False, False, True]


def test_band_validation_fires(tmp_path):
    bad = pd.DataFrame({
        "date": ["2020-01-01"],
        "tickers": [",".join(f"T{i}" for i in range(100))],  # 100 << band
    })
    p = tmp_path / "m.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="outside plausible band"):
        load_membership(csv_path=p, start="2019-01-01")
