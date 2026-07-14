"""Synthetic tests for scripts/fetch_congress_trades.py normalization (loaded
by path — scripts/ is not a package; no network is touched at import).

Covers: ticker mapping to the panel vocabulary, amount-range label parsing,
asset classification (non-equity filtering, options kept + flagged), owner and
type normalization, cross-filing dedup keeping the EARLIEST disclosure (the
PIT availability date), same-filing repeated lots preserved, and the
disclosure-lag computation inside build_table.
"""

import importlib.util
from pathlib import Path

import pandas as pd

_SCRIPT = (Path(__file__).resolve().parents[1] / "scripts"
           / "fetch_congress_trades.py")

_spec = importlib.util.spec_from_file_location("fetch_congress_trades", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# ---------------------------------------------------------------- unit pieces
def test_normalize_ticker():
    assert mod.normalize_ticker("BRK.B") == "BRK-B"      # panel vocabulary
    assert mod.normalize_ticker("bf.b ") == "BF-B"
    assert mod.normalize_ticker(" aapl") == "AAPL"
    assert mod.normalize_ticker("--") is None
    assert mod.normalize_ticker("") is None
    assert mod.normalize_ticker("N/A") is None
    assert mod.normalize_ticker(None) is None
    assert mod.normalize_ticker(float("nan")) is None
    assert mod.normalize_ticker("91282CLC3") is None     # treasury CUSIP
    assert mod.normalize_ticker("BRK B") is None         # embedded space = junk


def test_parse_amount_label():
    assert mod.parse_amount_label("$1,001 - $15,000") == (1001, 15000)
    assert mod.parse_amount_label("$1,000,001 - $5,000,000") == (1000001, 5000000)
    assert mod.parse_amount_label("Over $50,000,000") == (50000001, None)
    assert mod.parse_amount_label("$1,000,000 +") == (1000001, None)
    assert mod.parse_amount_label("Spouse/DC Over $1,000,000") == (1000001, None)
    assert mod.parse_amount_label(None) == (None, None)
    assert mod.parse_amount_label("unknown") == (None, None)


def test_classify_asset():
    # explicit codes/labels win
    assert mod.classify_asset("ST", "Apple Inc", "AAPL") == ("equity", False)
    assert mod.classify_asset("Stock", "Apple Inc", "AAPL") == ("equity", False)
    assert mod.classify_asset("OP", "Apple Inc", "AAPL") == ("option", False)
    assert mod.classify_asset("Stock Option", "x", None) == ("option", False)
    assert mod.classify_asset("Corporate Bond", "x", None) == ("non_equity", False)
    assert mod.classify_asset("Municipal Security", "x", None) == ("non_equity", False)
    # [XX] tag in the description, OCR-cased
    assert mod.classify_asset(None, "amazon.com, Inc. (aMZN) [sT]", "AMZN") == ("equity", False)
    assert mod.classify_asset(None, "Foo call spread [oP]", "FOO") == ("option", False)
    assert mod.classify_asset(None, "US Treasury 2% [gS]", None) == ("non_equity", False)
    # description keywords when untyped/untagged
    assert mod.classify_asset(
        None, "DEsCRIPTIoN: PuT Puma Biotechnology $275 Exp 12/20/14", "PBYI"
    ) == ("option", False)
    assert mod.classify_asset(None, "US Treasury bill", None) == ("non_equity", False)
    # bare ticker -> equity, flagged inferred
    assert mod.classify_asset(None, "Acme Corp", "ACME") == ("equity", True)
    assert mod.classify_asset(None, None, None) == ("unclassified", False)


def test_normalize_owner():
    assert mod.normalize_owner("SP") == "spouse"
    assert mod.normalize_owner("Spouse") == "spouse"
    assert mod.normalize_owner("JT") == "joint"
    assert mod.normalize_owner("Joint") == "joint"
    assert mod.normalize_owner("DC") == "child"
    assert mod.normalize_owner("Child") == "child"
    assert mod.normalize_owner("Self") == "self"
    assert mod.normalize_owner(None) == "undisclosed"    # never assume self
    assert mod.normalize_owner("SA") == "undisclosed"    # source parse noise


def test_normalize_type():
    assert mod.normalize_type("Purchase") == "purchase"
    assert mod.normalize_type("Sale (Full)") == "sale"
    assert mod.normalize_type("Sale (Partial)") == "sale"
    assert mod.normalize_type("Exchange") == "exchange"
    assert mod.normalize_type(None) == "other"


# --------------------------------------------------------------------- dedup
def _dedup_frame(rows):
    base = {"member_id": "house_a", "transaction_date": pd.Timestamp("2021-01-05"),
            "ticker": "AAPL", "asset_description": "Apple Inc (AAPL) [ST]",
            "type": "purchase", "amount_min": 1001, "amount_max": 15000,
            "owner": "spouse"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_dedup_collapses_refiling_to_earliest_disclosure():
    df = _dedup_frame([
        {"filing_id": "f1", "disclosure_date": pd.Timestamp("2021-02-01")},
        {"filing_id": "f2", "disclosure_date": pd.Timestamp("2021-03-15")},  # amendment re-list
    ])
    out = mod.dedup_trades(df)
    assert len(out) == 1
    assert out["disclosure_date"].iloc[0] == pd.Timestamp("2021-02-01")  # PIT: first public
    assert bool(out["is_refiled"].iloc[0])


def test_dedup_keeps_repeated_lots_within_one_filing():
    df = _dedup_frame([
        {"filing_id": "f1", "disclosure_date": pd.Timestamp("2021-02-01")},
        {"filing_id": "f1", "disclosure_date": pd.Timestamp("2021-02-01")},  # 2nd lot, same form
    ])
    out = mod.dedup_trades(df)
    assert len(out) == 2
    assert not out["is_refiled"].any()


def test_dedup_distinct_amounts_not_collapsed():
    df = _dedup_frame([
        {"filing_id": "f1", "disclosure_date": pd.Timestamp("2021-02-01")},
        {"filing_id": "f2", "disclosure_date": pd.Timestamp("2021-03-01"),
         "amount_min": 15001, "amount_max": 50000},
    ])
    assert len(mod.dedup_trades(df)) == 2


# --------------------------------------------------------------- build_table
FILERS = {
    "house_a": {"id": "house_a", "full_name": "Alice Alpha", "chamber": "house",
                "party": "D", "state": "CA"},
    "senate_b": {"id": "senate_b", "full_name": "Bob Beta", "chamber": "senate",
                 "party": "R", "state": "TX"},
}


def _raw(**kw):
    row = {"id": "x", "filing_id": "f1", "filer_id": "house_a",
           "source_id": "house_clerk", "transaction_date": "2021-01-05",
           "notification_date": None, "filing_date": "2021-02-04",
           "owner": "SP", "ticker": "AAPL", "asset_name": "Apple Inc (AAPL) [ST]",
           "asset_type": None, "transaction_type": "Purchase",
           "amount_range_low": 1001.0, "amount_range_high": 15000.0,
           "amount_range_label": "$1,001 - $15,000",
           "doc_url": "https://disclosures-clerk.house.gov/x.pdf",
           "filing_type": "PTR"}
    row.update(kw)
    return row


def test_build_table_end_to_end():
    trades = pd.DataFrame([
        _raw(id="a"),                                                  # kept, lag 30
        _raw(id="b", ticker="BRK.B", asset_name="Berkshire (BRK.B) [ST]",
             amount_range_low=None, amount_range_high=None),           # label fallback
        _raw(id="c", filer_id="senate_b", source_id="senate_efd", owner="Joint",
             asset_type="Stock Option", transaction_type="Sale (Partial)",
             filing_date="2021-01-05"),                                # option kept, lag 0
        _raw(id="d", asset_type="Corporate Bond", ticker=None),        # dropped: non-equity
        _raw(id="e", filing_type="278-T", filer_id="house_a"),         # dropped: OGE
        _raw(id="f", transaction_date=None),                           # dropped: no date
    ])
    out = mod.build_table(trades, FILERS)

    assert list(out.columns) == mod.COLUMNS
    assert len(out) == 3
    assert set(out["member"]) == {"Alice Alpha", "Bob Beta"}

    a = out[(out["ticker"] == "AAPL") & (out["member_id"] == "house_a")].iloc[0]
    assert a["disclosure_lag_days"] == 30
    assert a["owner"] == "spouse" and a["type"] == "purchase"
    assert not a["is_option"] and not a["asset_class_inferred"]
    assert a["chamber"] == "house" and a["party"] == "D" and a["state"] == "CA"

    b = out[out["ticker"] == "BRK-B"].iloc[0]                # dot -> dash
    assert b["amount_min"] == 1001 and b["amount_max"] == 15000  # parsed from label

    c = out[out["member_id"] == "senate_b"].iloc[0]
    assert c["is_option"] and c["type"] == "sale" and c["owner"] == "joint"
    assert c["disclosure_lag_days"] == 0                     # same-day filing is legal
    assert c["chamber"] == "senate"

    # dates are real datetimes, disclosure never before transaction here
    assert pd.api.types.is_datetime64_any_dtype(out["transaction_date"])
    assert pd.api.types.is_datetime64_any_dtype(out["disclosure_date"])
    assert (out["disclosure_lag_days"] >= 0).all()
