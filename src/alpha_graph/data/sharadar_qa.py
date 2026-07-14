"""Blind identity, membership, and price-coverage QA for Sharadar snapshots.

The audit is deliberately independent of factor returns. It operates on the
schema-validated Parquet snapshot produced by ``alpha_graph.data.sharadar``
and exits nonzero until both machine gates and manual identity cases pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from alpha_graph.config import CACHE_DIR, DATA_DIR, PROJECT_ROOT
from alpha_graph.data.pit_universe import (
    MEMBERSHIP_CSV,
    membership_asof,
)
from alpha_graph.data.sharadar import (
    RAW_ROOT,
    STAGED_ROOT,
    TABLE_SPECS,
    SharadarError,
    sha256_file,
    snapshot_lock,
    validate_snapshot_id,
    verify_snapshot_manifest,
)

DEFAULT_THRESHOLDS = PROJECT_ROOT / "config" / "vendor_qa_v1.json"
DEFAULT_CASES = PROJECT_ROOT / "config" / "identity_cases_v1.json"
REQUIRED_OVERRIDE_COLUMNS = {
    "reference_ticker",
    "reference_from",
    "reference_to",
    "vendor_ticker",
    "vendor_id",
    "status",
    "evidence",
    "snapshot",
    "tickers_sha256",
    "membership_sha256",
}
REQUIRED_APPROVAL_COLUMNS = {
    "case_id",
    "status",
    "evidence",
    "snapshot",
    "observed_ids_sha256",
    "tickers_sha256",
    "cases_sha256",
}
REQUIRED_IDENTITY_REGISTRY_COLUMNS = {
    "vendor_id",
    "listing_id",
    "issuer_id",
    "status",
    "evidence",
    "snapshot",
    "tickers_sha256",
}
REQUIRED_MEMBERSHIP_APPROVAL_COLUMNS = {
    "difference_id",
    "status",
    "evidence",
    "snapshot",
    "observed_sha256",
    "tickers_sha256",
    "membership_sha256",
}
REQUIRED_JUMP_APPROVAL_COLUMNS = {
    "jump_id",
    "status",
    "evidence",
    "snapshot",
    "observed_sha256",
    "tickers_sha256",
}
PRICE_NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "closeadj",
    "closeunadj",
    "volume",
    "dividends",
)
POLICY_INPUT_KEYS = (
    "thresholds",
    "cases",
    "overrides",
    "known_case_approvals",
    "membership_approvals",
    "identity_registry",
    "jump_approvals",
)


def _nonblank(value: object) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def _ticker_norm(value: object) -> str:
    return str(value).strip().upper().replace(".", "-")


def _vendor_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_thresholds(value: dict) -> dict:
    rates = {
        "departed_identity_coverage_min",
        "departed_interval_price_coverage_min",
        "member_day_price_coverage_min",
        "member_day_price_coverage_by_year_min",
        "membership_month_end_jaccard_min",
        "known_case_approval_rate_min",
        "identity_registry_coverage_min",
        "calendar_weekday_coverage_min",
        "calendar_month_weekday_coverage_min",
        "membership_difference_approval_rate_min",
    }
    counts = {
        "cross_section_min",
        "cross_section_max",
        "unresolved_primary_key_duplicates_max",
        "unresolved_identity_ambiguities_max",
        "calendar_endpoint_gap_days_max",
        "calendar_internal_gap_days_max",
        "calendar_expected_dates",
        "sep_quarantined_rows_max",
    }
    missing = sorted(({"schema_version"} | rates | counts) - set(value))
    if missing:
        raise SharadarError(f"QA thresholds missing keys: {missing}")
    if value["schema_version"] != 1:
        raise SharadarError("unsupported QA threshold schema_version")
    for key in rates:
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not 0 <= raw <= 1:
            raise SharadarError(f"QA threshold {key} must be a number in [0, 1]")
    for key in counts:
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise SharadarError(f"QA threshold {key} must be a nonnegative integer")
    if value["cross_section_min"] > value["cross_section_max"]:
        raise SharadarError("cross_section_min exceeds cross_section_max")
    calendar_hash = value.get("calendar_expected_sha256")
    if (
        not isinstance(calendar_hash, str)
        or len(calendar_hash) != 64
        or any(character not in "0123456789abcdef" for character in calendar_hash)
    ):
        raise SharadarError("calendar_expected_sha256 must be a lowercase SHA-256 digest")
    return value


def validate_cases_config(value: dict) -> dict:
    if value.get("schema_version") != 1 or not isinstance(value.get("cases"), list):
        raise SharadarError("identity cases must use schema_version 1 and contain a list")
    case_ids = []
    for case in value["cases"]:
        if not isinstance(case, dict):
            raise SharadarError("identity case entries must be objects")
        missing = {"case_id", "symbols", "expectation"} - set(case)
        if missing or not _nonblank(case.get("case_id")):
            raise SharadarError(f"identity case is missing fields: {sorted(missing)}")
        if not isinstance(case["symbols"], list) or not case["symbols"]:
            raise SharadarError(f"identity case {case['case_id']} has no symbols")
        normalized_symbols = [_ticker_norm(symbol) for symbol in case["symbols"]]
        if (
            any(not _nonblank(symbol) for symbol in case["symbols"])
            or len(normalized_symbols) != len(set(normalized_symbols))
            or not _nonblank(case["expectation"])
        ):
            raise SharadarError(
                f"identity case {case['case_id']} has blank or duplicate content"
            )
        case_ids.append(str(case["case_id"]))
    if len(case_ids) != len(set(case_ids)):
        raise SharadarError("identity cases contain duplicate case_id values")
    return value


def load_reference_snapshots_raw(
    csv_path: Path,
    *,
    xs_band: tuple[int, int] = (490, 515),
) -> list[tuple[pd.Timestamp, frozenset[str]]]:
    """Load source-space membership without applying the panel rename map."""
    frame = pd.read_csv(csv_path, usecols=["date", "tickers"])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.empty:
        raise SharadarError("reference membership is empty")
    if not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise SharadarError("reference membership dates are not strictly increasing")
    frame = frame.reset_index(drop=True)

    snapshots: list[tuple[pd.Timestamp, frozenset[str]]] = []
    for row in frame.itertuples(index=False):
        raw = [_ticker_norm(value) for value in str(row.tickers).split(",")]
        symbols = [value for value in raw if value]
        if len(symbols) != len(set(symbols)):
            raise SharadarError(
                f"reference snapshot {row.date.date()} contains duplicate symbols"
            )
        if not (xs_band[0] <= len(symbols) <= xs_band[1]):
            raise SharadarError(
                f"reference snapshot {row.date.date()} cross-section {len(symbols)} "
                f"outside [{xs_band[0]}, {xs_band[1]}]"
            )
        snapshots.append((pd.Timestamp(row.date), frozenset(symbols)))
    return snapshots


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_worktree() -> dict:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"dirty": None, "dirty_path_count": None}
    entries = [line for line in result.stdout.splitlines() if line.strip()]
    return {"dirty": bool(entries), "dirty_path_count": len(entries)}


def read_staged_table(
    snapshot_dir: Path,
    table: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    table_dir = snapshot_dir / table.upper()
    if table_dir.is_symlink() or (
        table_dir.exists() and table_dir.resolve().parent != snapshot_dir.resolve()
    ):
        raise SharadarError(f"unsafe staged table directory {table_dir}")
    files = sorted(table_dir.glob("*.parquet"))
    if not files:
        raise SharadarError(f"no staged {table.upper()} Parquet files in {snapshot_dir}")
    return pd.read_parquet(files, columns=columns)


def staged_table_sha256(snapshot_dir: Path, table: str) -> str:
    files = sorted((snapshot_dir / table.upper()).glob("*.parquet"))
    if not files:
        raise SharadarError(f"no staged {table.upper()} files to hash")
    inventory = [
        {"name": path.name, "sha256": sha256_file(path)}
        for path in files
    ]
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def membership_intervals(
    snapshots: list[tuple[pd.Timestamp, frozenset]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Convert point-in-time snapshots to inclusive ticker membership spells."""
    active: dict[str, pd.Timestamp] = {}
    previous: set[str] = set()
    rows: list[dict] = []
    for snapshot_date, members_frozen in snapshots:
        if snapshot_date > end:
            break
        members = set(members_frozen)
        for ticker in previous - members:
            rows.append({
                "reference_ticker": ticker,
                "valid_from": active.pop(ticker),
                "valid_to": snapshot_date - pd.Timedelta(days=1),
            })
        for ticker in members - previous:
            active[ticker] = snapshot_date
        previous = members
    for ticker, valid_from in active.items():
        rows.append({
            "reference_ticker": ticker,
            "valid_from": valid_from,
            "valid_to": end,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SharadarError("reference membership produced no intervals")
    frame["valid_from"] = frame["valid_from"].clip(lower=start)
    frame["valid_to"] = frame["valid_to"].clip(upper=end)
    frame = frame[frame["valid_from"] <= frame["valid_to"]].copy()
    frame["reference_ticker"] = frame["reference_ticker"].map(_ticker_norm)
    frame = frame.sort_values(
        ["reference_ticker", "valid_from", "valid_to"]
    ).reset_index(drop=True)
    frame["interval_id"] = [
        f"{row.reference_ticker}:{row.valid_from:%Y%m%d}:{row.valid_to:%Y%m%d}"
        for row in frame.itertuples()
    ]
    return frame


def prepare_vendor_tickers(frame: pd.DataFrame) -> pd.DataFrame:
    required = TABLE_SPECS["TICKERS"].required_columns
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SharadarError(f"TICKERS missing required columns: {missing}")
    out = frame.copy()
    out = out[out["table"].astype(str).str.upper().str.endswith("SEP")].copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["ticker_norm"] = out["ticker"].map(_ticker_norm)
    out["vendor_id"] = out["permaticker"].map(_vendor_id)
    out["firstpricedate"] = pd.to_datetime(out["firstpricedate"], errors="coerce")
    out["lastpricedate"] = pd.to_datetime(out["lastpricedate"], errors="coerce")
    out["bounds_complete"] = (
        out["firstpricedate"].notna() & out["lastpricedate"].notna()
    )
    return out


def load_identity_overrides(
    path: Path | None,
    *,
    snapshot: str | None = None,
    tickers_sha256: str | None = None,
    membership_sha256: str | None = None,
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=sorted(REQUIRED_OVERRIDE_COLUMNS))
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_OVERRIDE_COLUMNS - set(frame.columns))
    if missing:
        raise SharadarError(f"identity overrides missing columns: {missing}")
    frame = frame.copy()
    frame["reference_ticker"] = frame["reference_ticker"].map(_ticker_norm)
    frame["vendor_ticker"] = frame["vendor_ticker"].astype(str).str.strip().str.upper()
    frame["reference_from"] = pd.to_datetime(frame["reference_from"])
    frame["reference_to"] = pd.to_datetime(frame["reference_to"])
    frame["binding_valid"] = (
        frame["snapshot"].astype(str).eq(snapshot)
        & frame["tickers_sha256"].astype(str).eq(tickers_sha256)
        & frame["membership_sha256"].astype(str).eq(membership_sha256)
    )
    return frame


def resolve_identity_intervals(
    intervals: pd.DataFrame,
    vendor_tickers: pd.DataFrame,
    overrides: pd.DataFrame | None = None,
    *,
    tolerance_days: int = 7,
) -> pd.DataFrame:
    overrides = overrides if overrides is not None else load_identity_overrides(None)
    by_ticker = {key: group for key, group in vendor_tickers.groupby("ticker_norm", sort=False)}
    tolerance = pd.Timedelta(days=tolerance_days)
    rows = []
    for interval in intervals.itertuples(index=False):
        binding_valid = (
            overrides["binding_valid"]
            if "binding_valid" in overrides
            else pd.Series(False, index=overrides.index)
        )
        manual = overrides[
            (overrides["reference_ticker"] == interval.reference_ticker)
            & (overrides["reference_from"] <= interval.valid_from)
            & (overrides["reference_to"] >= interval.valid_to)
            & (overrides["status"].astype(str).str.lower() == "approved")
            & overrides["evidence"].map(_nonblank)
            & binding_valid
        ]
        method = "exact_ticker_interval"
        evidence = ""
        if len(manual) > 1:
            candidates = pd.DataFrame()
            status = "ambiguous_override"
        elif len(manual) == 1:
            override = manual.iloc[0]
            wanted_id = _vendor_id(override["vendor_id"])
            candidates = vendor_tickers[
                vendor_tickers["ticker_norm"] == _ticker_norm(override["vendor_ticker"])
            ]
            if wanted_id:
                candidates = candidates[candidates["vendor_id"] == wanted_id]
            method = "approved_override"
            evidence = str(override["evidence"])
            status = "pending"
        else:
            candidates = by_ticker.get(
                interval.reference_ticker, vendor_tickers.iloc[0:0]
            ).copy()
            status = "pending"

        if status == "pending":
            candidates = candidates[
                candidates["bounds_complete"]
                & (candidates["firstpricedate"] <= interval.valid_from + tolerance)
                & (candidates["lastpricedate"] >= interval.valid_to - tolerance)
            ]
            ids = sorted({value for value in candidates["vendor_id"] if value})
            if len(ids) == 1:
                status = "resolved"
                vendor_id = ids[0]
                ticker_values = sorted(
                    candidates.loc[candidates["vendor_id"] == vendor_id, "ticker"].unique()
                )
                vendor_ticker = ticker_values[0]
            elif not ids:
                status = "unmatched"
                vendor_id = None
                vendor_ticker = None
            else:
                status = "ambiguous"
                vendor_id = None
                vendor_ticker = None
        else:
            ids = []
            vendor_id = None
            vendor_ticker = None
        rows.append({
            **interval._asdict(),
            "vendor_ticker": vendor_ticker,
            "vendor_id": vendor_id,
            "candidate_vendor_ids": "|".join(ids),
            "candidate_count": len(ids),
            "match_method": method,
            "match_status": status,
            "evidence": evidence,
            "listing_id": None,
            "issuer_id": None,
            "vendor_id_semantics": "unverified; identity registry required",
        })
    return pd.DataFrame(rows)


def _valid_price_rows(prices: pd.DataFrame) -> pd.DataFrame:
    """Return price rows whose numeric fields are all present and finite."""
    required = {"ticker", "date", *PRICE_NUMERIC_COLUMNS}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise SharadarError(f"SEP price coverage missing columns: {missing}")
    frame = prices[["ticker", "date", *PRICE_NUMERIC_COLUMNS]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    numeric = frame[list(PRICE_NUMERIC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)
    return frame[valid]


def _price_dates_by_ticker(prices: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    frame = _valid_price_rows(prices)
    return {
        ticker: pd.DatetimeIndex(group["date"].drop_duplicates().sort_values())
        for ticker, group in frame.groupby("ticker", sort=False)
    }


def _price_dates_by_listing(
    prices: pd.DataFrame,
    row_resolution: pd.DataFrame,
) -> dict[str, pd.DatetimeIndex]:
    """Group numeric-valid observed dates by resolved listing key."""
    frame = _valid_price_rows(prices)
    frame["ticker_norm"] = frame["ticker"].map(_ticker_norm)
    pairs = frame[["ticker_norm", "date"]].drop_duplicates()
    resolved = row_resolution[["ticker_norm", "date", "listing_key"]].drop_duplicates()
    merged = pairs.merge(resolved, on=["ticker_norm", "date"], how="inner")
    return {
        key: pd.DatetimeIndex(group["date"].drop_duplicates().sort_values())
        for key, group in merged.groupby("listing_key", sort=False)
    }


def resolve_sep_rows(
    prices: pd.DataFrame,
    vendor_tickers: pd.DataFrame,
    *,
    identity_map: dict[str, str] | None = None,
    tolerance_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve each SEP (ticker, date) row to exactly one listing window.

    A row matches a TICKERS window when the normalized tickers agree and the
    row date lies inside [firstpricedate - tolerance, lastpricedate +
    tolerance]. Vendor IDs collapse to listing IDs through the approved
    identity registry where available. Rows matching zero or multiple
    listings are quarantined and must never enter coverage numerators.
    """
    missing = sorted({"ticker", "date"} - set(prices.columns))
    if missing:
        raise SharadarError(f"SEP row resolution missing columns: {missing}")
    rows = prices[["ticker", "date"]].copy()
    rows["ticker_norm"] = rows["ticker"].map(_ticker_norm)
    rows["date"] = pd.to_datetime(rows["date"], errors="raise")
    mapping = identity_map or {}
    windows = vendor_tickers.loc[
        vendor_tickers["bounds_complete"] & vendor_tickers["vendor_id"].map(_nonblank),
        ["ticker_norm", "vendor_id", "firstpricedate", "lastpricedate"],
    ].copy()
    windows["listing_key"] = windows["vendor_id"].map(
        lambda value: mapping.get(str(value), f"unverified:{value}")
    )
    tolerance = pd.Timedelta(days=tolerance_days)
    pairs = rows[["ticker_norm", "date"]].drop_duplicates()
    merged = pairs.merge(windows, on="ticker_norm", how="inner")
    merged = merged[
        (merged["firstpricedate"] <= merged["date"] + tolerance)
        & (merged["lastpricedate"] >= merged["date"] - tolerance)
    ]
    matches = merged.drop_duplicates(["ticker_norm", "date", "listing_key"]).groupby(
        ["ticker_norm", "date"], as_index=False
    ).agg(
        listing_count=("listing_key", "nunique"),
        listing_key=("listing_key", "first"),
        candidate_listings=("listing_key", lambda values: "|".join(sorted(values))),
    )
    out = rows.merge(matches, on=["ticker_norm", "date"], how="left")
    out["listing_count"] = out["listing_count"].fillna(0).astype(int)
    out["candidate_listings"] = out["candidate_listings"].fillna("")
    resolved = out.loc[
        out["listing_count"] == 1,
        ["ticker", "ticker_norm", "date", "listing_key"],
    ].copy()
    quarantined = out.loc[
        out["listing_count"] != 1,
        ["ticker", "ticker_norm", "date", "listing_count", "candidate_listings"],
    ].copy()
    quarantined["quarantine_reason"] = np.where(
        quarantined["listing_count"] == 0, "zero_listings", "multiple_listings"
    )
    return resolved.reset_index(drop=True), quarantined.reset_index(drop=True)


def price_coverage(
    crosswalk: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    row_resolution: pd.DataFrame | None = None,
    identity_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if row_resolution is None:
        observed_by_key = _price_dates_by_ticker(prices)

        def interval_key(row) -> str | None:
            return row.vendor_ticker
    else:
        observed_by_key = _price_dates_by_listing(prices, row_resolution)
        mapping = identity_map or {}

        def interval_key(row) -> str | None:
            if not _nonblank(row.vendor_id):
                return None
            return mapping.get(str(row.vendor_id), f"unverified:{row.vendor_id}")

    interval_rows = []
    year_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for row in crosswalk.itertuples(index=False):
        expected = calendar[(calendar >= row.valid_from) & (calendar <= row.valid_to)]
        observed_dates = observed_by_key.get(interval_key(row), pd.DatetimeIndex([]))
        observed = expected.isin(observed_dates) if len(expected) else np.array([], dtype=bool)
        expected_n = int(len(expected))
        observed_n = int(observed.sum())
        interval_rows.append({
            "interval_id": row.interval_id,
            "reference_ticker": row.reference_ticker,
            "vendor_ticker": row.vendor_ticker,
            "vendor_id": row.vendor_id,
            "match_status": row.match_status,
            "expected_member_days": expected_n,
            "observed_price_days": observed_n,
            "coverage": observed_n / expected_n if expected_n else 0.0,
        })
        for year in sorted(set(expected.year)):
            mask = expected.year == year
            year_counts[int(year)][0] += int(mask.sum())
            year_counts[int(year)][1] += int(observed[mask].sum())
    by_interval = pd.DataFrame(interval_rows)
    by_year = pd.DataFrame([
        {
            "year": year,
            "expected_member_days": counts[0],
            "observed_price_days": counts[1],
            "coverage": counts[1] / counts[0] if counts[0] else np.nan,
        }
        for year, counts in sorted(year_counts.items())
    ])
    expected_total = int(by_interval["expected_member_days"].sum())
    observed_total = int(by_interval["observed_price_days"].sum())
    summary = {
        "expected_member_days": expected_total,
        "observed_price_days": observed_total,
        "coverage": observed_total / expected_total if expected_total else 0.0,
        "min_year_coverage": float(by_year["coverage"].min()) if len(by_year) else 0.0,
    }
    return by_interval, by_year, summary


def resolve_panel_identities(
    panel: pd.DataFrame,
    vendor_tickers: pd.DataFrame,
    *,
    tolerance_days: int = 7,
    as_of: pd.Timestamp | None = None,
    recency_days: int = 7,
) -> pd.DataFrame:
    """Resolve back-normalized panel symbols at each series' final observation."""
    frame = panel[["ticker", "date"]].copy()
    frame["ticker"] = frame["ticker"].map(_ticker_norm)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    bounds = frame.groupby("ticker", as_index=False).agg(
        first_panel_date=("date", "min"),
        last_panel_date=("date", "max"),
    )
    tolerance = pd.Timedelta(days=tolerance_days)
    rows = []
    for row in bounds.itertuples(index=False):
        candidates = vendor_tickers[
            (vendor_tickers["ticker_norm"] == row.ticker)
            & vendor_tickers["bounds_complete"]
            & (vendor_tickers["firstpricedate"] <= row.last_panel_date + tolerance)
            & (vendor_tickers["lastpricedate"] >= row.last_panel_date - tolerance)
        ]
        ids = sorted({value for value in candidates["vendor_id"] if value})
        status = "resolved" if len(ids) == 1 else ("unmatched" if not ids else "ambiguous")
        rows.append({
            "panel_ticker": row.ticker,
            "first_panel_date": row.first_panel_date,
            "last_panel_date": row.last_panel_date,
            "vendor_id": ids[0] if len(ids) == 1 else None,
            "candidate_vendor_ids": "|".join(ids),
            "match_status": status,
            "current_at_audit_end": (
                True if as_of is None
                else row.last_panel_date >= as_of - pd.Timedelta(days=recency_days)
            ),
        })
    return pd.DataFrame(rows)


def calendar_quality(
    calendar: pd.DatetimeIndex,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    if calendar.empty:
        raise SharadarError("independent calendar is empty")
    calendar = pd.DatetimeIndex(calendar).drop_duplicates().sort_values()
    weekday_count = len(pd.bdate_range(start, end))
    month_coverages = []
    for period in pd.period_range(start, end, freq="M"):
        month_start = max(start, period.start_time)
        month_end = min(end, period.end_time.normalize())
        month_weekdays = len(pd.bdate_range(month_start, month_end))
        observed = int((calendar.to_period("M") == period).sum())
        month_coverages.append(observed / month_weekdays if month_weekdays else 1.0)
    internal_gap = int(
        pd.Series(calendar).sort_values().diff().dt.days.max()
    ) if len(calendar) > 1 else 0
    return {
        "weekday_coverage": len(calendar) / weekday_count if weekday_count else 0.0,
        "min_month_weekday_coverage": min(month_coverages) if month_coverages else 0.0,
        "start_gap_days": max(0, int((calendar.min() - start).days)),
        "end_gap_days": max(0, int((end - calendar.max()).days)),
        "max_internal_gap_days": internal_gap,
        "sha256": hashlib.sha256(
            ("\n".join(value.strftime("%Y-%m-%d") for value in calendar) + "\n")
            .encode("utf-8")
        ).hexdigest(),
    }


def departed_target_coverage(
    crosswalk: pd.DataFrame,
    coverage_by_interval: pd.DataFrame,
    panel_vendor_ids: set[str],
    *,
    interval_coverage_min: float = 0.98,
) -> tuple[pd.DataFrame, dict]:
    frame = crosswalk.merge(
        coverage_by_interval[[
            "interval_id", "expected_member_days", "observed_price_days", "coverage"
        ]],
        on="interval_id",
        how="left",
        validate="one_to_one",
    )
    resolved = frame["match_status"].eq("resolved") & frame["vendor_id"].notna()
    frame["identity_key"] = np.where(
        resolved,
        "vendor:" + frame["vendor_id"].fillna(""),
        "unresolved:" + frame["interval_id"],
    )
    frame["is_departed_target"] = ~resolved | ~frame["vendor_id"].isin(panel_vendor_ids)
    departed = frame[frame["is_departed_target"]].copy()
    if departed.empty:
        status = pd.DataFrame(columns=[
            "identity_key",
            "reference_tickers",
            "vendor_id",
            "intervals",
            "all_resolved",
            "all_intervals_expected",
            "min_interval_coverage",
            "all_intervals_covered",
            "covered",
        ])
        return status, {
            "denominator_kind": (
                "unique listing identities absent from the current panel; "
                "unresolved reference intervals remain failed targets"
            ),
            "interval_coverage_min": interval_coverage_min,
            "target": 0,
            "covered": 0,
            "coverage": 1.0,
        }
    status = departed.groupby("identity_key", as_index=False).agg(
        reference_tickers=("reference_ticker", lambda values: "|".join(sorted(set(values)))),
        vendor_id=("vendor_id", "first"),
        intervals=("interval_id", "size"),
        all_resolved=("match_status", lambda values: bool((values == "resolved").all())),
        all_intervals_expected=(
            "expected_member_days", lambda values: bool((values > 0).all())
        ),
        min_interval_coverage=("coverage", "min"),
    )
    status["all_intervals_covered"] = (
        status["min_interval_coverage"].fillna(0.0) >= interval_coverage_min
    )
    status["covered"] = (
        status["all_resolved"]
        & status["all_intervals_expected"]
        & status["all_intervals_covered"]
    )
    target = int(len(status))
    covered = int(status["covered"].sum())
    return status, {
        "denominator_kind": (
            "unique listing identities absent from the current panel; "
            "unresolved reference intervals remain failed targets"
        ),
        "interval_coverage_min": interval_coverage_min,
        "target": target,
        "covered": covered,
        "coverage": covered / target if target else 1.0,
    }


def _resolve_symbol_ids(
    symbols: set[str],
    when: pd.Timestamp,
    vendor_tickers: pd.DataFrame,
    *,
    tolerance_days: int = 7,
    identity_map: dict[str, str] | None = None,
) -> tuple[set[str], int]:
    ids: set[str] = set()
    unresolved = 0
    tolerance = pd.Timedelta(days=tolerance_days)
    for symbol in symbols:
        norm = _ticker_norm(symbol)
        candidates = vendor_tickers[
            (vendor_tickers["ticker_norm"] == norm)
            & (vendor_tickers["firstpricedate"] <= when + tolerance)
            & (vendor_tickers["lastpricedate"] >= when - tolerance)
        ]
        values = {value for value in candidates["vendor_id"] if value}
        if len(values) == 1:
            value = next(iter(values))
            if identity_map is not None:
                value = identity_map.get(value, f"unverified:{value}")
            if value in ids:
                unresolved += 1
            else:
                ids.add(value)
        else:
            unresolved += 1
    return ids, unresolved


def membership_reconciliation(
    sp500: pd.DataFrame,
    reference_snapshots: list[tuple[pd.Timestamp, frozenset]],
    vendor_tickers: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    comparison_dates: pd.DatetimeIndex | None = None,
    identity_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    frame = sp500.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = frame["ticker"].map(_ticker_norm)
    frame["action_norm"] = frame["action"].astype(str).str.strip().str.lower()
    current_rows = frame[frame["action_norm"] == "current"]
    state_symbols = set(current_rows["ticker"])
    if not state_symbols:
        raise SharadarError("SP500 table has no current rows; cannot reconstruct membership")
    anchor_date = current_rows["date"].max()
    if anchor_date < end:
        raise SharadarError(
            f"SP500 current-state anchor {anchor_date.date()} precedes audit end {end.date()}"
        )
    events = frame[frame["action_norm"].isin({"added", "removed"})].copy()
    events = events.sort_values("date", ascending=False)
    if comparison_dates is None:
        dates = list(pd.date_range(start=start, end=end, freq="ME"))
        if end not in dates:
            dates.append(end)
    else:
        dates = [pd.Timestamp(value) for value in comparison_dates
                 if start <= pd.Timestamp(value) <= end]
    checkpoints = pd.DatetimeIndex(sorted(set(dates), reverse=True))

    event_rows = list(events.itertuples(index=False))
    event_index = 0
    rows = []
    for when in checkpoints:
        while event_index < len(event_rows) and event_rows[event_index].date > when:
            event = event_rows[event_index]
            if event.action_norm == "added":
                state_symbols.discard(event.ticker)
            else:
                state_symbols.add(event.ticker)
            event_index += 1
        reference = membership_asof(reference_snapshots, when)
        if reference is None:
            continue
        reference_ids, unresolved_reference = _resolve_symbol_ids(
            set(reference), when, vendor_tickers, identity_map=identity_map
        )
        vendor_ids, unresolved_vendor = _resolve_symbol_ids(
            state_symbols, when, vendor_tickers, identity_map=identity_map
        )
        union = reference_ids | vendor_ids
        denominator = len(union) + unresolved_reference + unresolved_vendor
        intersection = len(reference_ids & vendor_ids)
        jaccard = intersection / denominator if denominator else 0.0
        rows.append({
            "date": when,
            "reference_count": len(reference),
            "vendor_raw_count": len(state_symbols),
            "reference_vendor_id_count": len(reference_ids),
            "vendor_id_count": len(vendor_ids),
            "unresolved_reference": unresolved_reference,
            "unresolved_vendor": unresolved_vendor,
            "reference_only_ids": "|".join(sorted(reference_ids - vendor_ids)),
            "vendor_only_ids": "|".join(sorted(vendor_ids - reference_ids)),
            "jaccard": jaccard,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def membership_difference_report(
    daily: pd.DataFrame,
    approvals_path: Path | None,
    *,
    snapshot: str,
    tickers_sha256: str,
    membership_sha256: str,
    minimum_trading_days: int = 5,
) -> pd.DataFrame:
    """Build and adjudicate resolved membership-difference spells."""
    approvals = pd.DataFrame(columns=sorted(REQUIRED_MEMBERSHIP_APPROVAL_COLUMNS))
    if approvals_path is not None:
        approvals = pd.read_csv(approvals_path, dtype=str, keep_default_na=False)
        missing = REQUIRED_MEMBERSHIP_APPROVAL_COLUMNS - set(approvals.columns)
        if missing:
            raise SharadarError(
                f"membership approvals missing columns: {sorted(missing)}"
            )
        if approvals["difference_id"].duplicated().any():
            raise SharadarError("membership approvals contain duplicate difference_id values")
    approval_map = (
        approvals.set_index("difference_id").to_dict("index")
        if len(approvals) else {}
    )

    active: dict[tuple[str, str], dict] = {}
    completed: list[dict] = []
    ordered = daily.sort_values("date").reset_index(drop=True)
    for _, row in ordered.iterrows():
        present = set()
        for column, side in (
            ("reference_only_ids", "reference_only"),
            ("vendor_only_ids", "vendor_only"),
        ):
            present.update(
                (side, value) for value in str(row[column]).split("|") if value
            )
        for key in list(active):
            if key not in present:
                completed.append(active.pop(key))
        for key in present:
            if key not in active:
                active[key] = {
                    "side": key[0],
                    "vendor_id": key[1],
                    "start_date": row["date"],
                    "end_date": row["date"],
                    "trading_days": 1,
                }
            else:
                active[key]["end_date"] = row["date"]
                active[key]["trading_days"] += 1
    completed.extend(active.values())

    rows = []
    for spell in completed:
        if spell["trading_days"] <= minimum_trading_days:
            continue
        observed = {
            "side": spell["side"],
            "vendor_id": spell["vendor_id"],
            "start_date": str(pd.Timestamp(spell["start_date"]).date()),
            "end_date": str(pd.Timestamp(spell["end_date"]).date()),
            "trading_days": spell["trading_days"],
        }
        serialized = json.dumps(observed, sort_keys=True, separators=(",", ":"))
        observed_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        difference_id = (
            f"{spell['side']}:{spell['vendor_id']}:"
            f"{observed['start_date']}:{observed['end_date']}"
        )
        approval = approval_map.get(difference_id, {})
        approved = (
            str(approval.get("status", "")).lower() == "approved"
            and _nonblank(approval.get("evidence"))
            and str(approval.get("snapshot", "")) == snapshot
            and str(approval.get("observed_sha256", "")) == observed_sha
            and str(approval.get("tickers_sha256", "")) == tickers_sha256
            and str(approval.get("membership_sha256", "")) == membership_sha256
        )
        rows.append({
            "difference_id": difference_id,
            **observed,
            "observed_sha256": observed_sha,
            "manual_status": "approved" if approved else "review_required",
            "evidence": (
                "" if pd.isna(approval.get("evidence"))
                else str(approval.get("evidence", ""))
            ),
        })
    return pd.DataFrame(rows, columns=[
        "difference_id",
        "side",
        "vendor_id",
        "start_date",
        "end_date",
        "trading_days",
        "observed_sha256",
        "manual_status",
        "evidence",
    ])


def known_case_report(
    vendor_tickers: pd.DataFrame,
    cases_config: dict,
    approvals_path: Path | None,
    *,
    snapshot: str,
    tickers_sha256: str,
    cases_sha256: str,
) -> pd.DataFrame:
    approvals = pd.DataFrame(columns=sorted(REQUIRED_APPROVAL_COLUMNS))
    if approvals_path is not None:
        approvals = pd.read_csv(approvals_path)
        missing = REQUIRED_APPROVAL_COLUMNS - set(approvals.columns)
        if missing:
            raise SharadarError(f"case approvals missing columns: {sorted(missing)}")
        if approvals["case_id"].duplicated().any():
            raise SharadarError("case approvals contain duplicate case_id values")
    approval_map = approvals.set_index("case_id").to_dict("index") if len(approvals) else {}
    rows = []
    for case in cases_config["cases"]:
        ids_by_symbol = {}
        for symbol in case["symbols"]:
            candidates = vendor_tickers[vendor_tickers["ticker_norm"] == _ticker_norm(symbol)]
            ids_by_symbol[symbol] = sorted({value for value in candidates["vendor_id"] if value})
        relevant = vendor_tickers[
            vendor_tickers["ticker_norm"].isin(
                {_ticker_norm(symbol) for symbol in case["symbols"]}
            )
        ][["table", "vendor_id", "ticker", "name", "firstpricedate", "lastpricedate"]].copy()
        for column in ("firstpricedate", "lastpricedate"):
            relevant[column] = relevant[column].map(
                lambda value: None if pd.isna(value) else str(pd.Timestamp(value).date())
            )
        relevant = relevant.sort_values(
            ["ticker", "vendor_id", "firstpricedate", "lastpricedate"],
            na_position="first",
        )
        observed_rows = relevant.to_dict("records")
        observed = json.dumps(observed_rows, sort_keys=True, separators=(",", ":"))
        observed_sha = hashlib.sha256(observed.encode("utf-8")).hexdigest()
        approval = approval_map.get(case["case_id"], {})
        approved = (
            str(approval.get("status", "")).lower() == "approved"
            and _nonblank(approval.get("evidence"))
            and str(approval.get("snapshot", "")) == snapshot
            and str(approval.get("observed_ids_sha256", "")) == observed_sha
            and str(approval.get("tickers_sha256", "")) == tickers_sha256
            and str(approval.get("cases_sha256", "")) == cases_sha256
        )
        if approved:
            approval_reason = "snapshot_and_observed_ids_match"
        elif not approval:
            approval_reason = "missing_approval"
        elif not _nonblank(approval.get("evidence")):
            approval_reason = "blank_evidence"
        elif str(approval.get("snapshot", "")) != snapshot:
            approval_reason = "snapshot_mismatch"
        elif str(approval.get("observed_ids_sha256", "")) != observed_sha:
            approval_reason = "observed_ids_hash_mismatch"
        elif str(approval.get("tickers_sha256", "")) != tickers_sha256:
            approval_reason = "tickers_hash_mismatch"
        elif str(approval.get("cases_sha256", "")) != cases_sha256:
            approval_reason = "cases_hash_mismatch"
        else:
            approval_reason = "status_not_approved"
        rows.append({
            "case_id": case["case_id"],
            "symbols": "|".join(case["symbols"]),
            "expectation": case["expectation"],
            "observed_vendor_ids": json.dumps(
                ids_by_symbol, sort_keys=True, separators=(",", ":")
            ),
            "observed_ticker_rows": observed,
            "observed_ids_sha256": observed_sha,
            "manual_status": "approved" if approved else "review_required",
            "approval_reason": approval_reason,
            "approval_snapshot": (
                "" if pd.isna(approval.get("snapshot")) else str(approval.get("snapshot", ""))
            ),
            "evidence": (
                "" if pd.isna(approval.get("evidence")) else str(approval.get("evidence", ""))
            ),
        })
    return pd.DataFrame(rows, columns=[
        "case_id",
        "symbols",
        "expectation",
        "observed_vendor_ids",
        "observed_ticker_rows",
        "observed_ids_sha256",
        "manual_status",
        "approval_reason",
        "approval_snapshot",
        "evidence",
    ])


def identity_registry_report(
    path: Path | None,
    *,
    snapshot: str,
    tickers_sha256: str,
    relevant_vendor_ids: set[str],
) -> tuple[pd.DataFrame, dict]:
    """Load snapshot-bound listing/issuer IDs; unbound rows never count."""
    if path is None:
        frame = pd.DataFrame(columns=sorted(REQUIRED_IDENTITY_REGISTRY_COLUMNS))
    else:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = REQUIRED_IDENTITY_REGISTRY_COLUMNS - set(frame.columns)
        if missing:
            raise SharadarError(f"identity registry missing columns: {sorted(missing)}")
        if frame["vendor_id"].duplicated().any():
            raise SharadarError("identity registry contains duplicate vendor_id values")
    frame = frame.copy()
    if len(frame):
        frame["vendor_id"] = frame["vendor_id"].map(_vendor_id)
        frame["bound_and_approved"] = (
            frame["status"].str.lower().eq("approved")
            & frame["snapshot"].eq(snapshot)
            & frame["tickers_sha256"].eq(tickers_sha256)
            & frame["listing_id"].map(_nonblank)
            & frame["issuer_id"].map(_nonblank)
            & frame["evidence"].map(_nonblank)
        )
    else:
        frame["bound_and_approved"] = pd.Series(dtype=bool)
    approved_ids = set(
        frame.loc[frame["bound_and_approved"], "vendor_id"].dropna().astype(str)
    )
    covered_ids = relevant_vendor_ids & approved_ids
    total = len(relevant_vendor_ids)
    summary = {
        "required_vendor_ids": total,
        "approved_vendor_ids": len(covered_ids),
        "coverage": len(covered_ids) / total if total else 1.0,
        "tickers_sha256": tickers_sha256,
        "status": "approved" if covered_ids == relevant_vendor_ids else "review_required",
    }
    return frame, summary


def alias_window_audit(vendor_tickers: pd.DataFrame) -> pd.DataFrame:
    """Report ticker aliases concurrently assigned to distinct vendor IDs."""
    rows = []
    complete = vendor_tickers[vendor_tickers["bounds_complete"]].copy()
    for ticker, group in complete.groupby("ticker_norm", sort=False):
        values = list(group.sort_values("firstpricedate").itertuples(index=False))
        for left_index, left in enumerate(values):
            for right in values[left_index + 1:]:
                if left.vendor_id == right.vendor_id:
                    continue
                overlap_from = max(left.firstpricedate, right.firstpricedate)
                overlap_to = min(left.lastpricedate, right.lastpricedate)
                if overlap_from <= overlap_to:
                    rows.append({
                        "ticker": ticker,
                        "left_vendor_id": left.vendor_id,
                        "right_vendor_id": right.vendor_id,
                        "overlap_from": overlap_from,
                        "overlap_to": overlap_to,
                    })
    return pd.DataFrame(rows, columns=[
        "ticker",
        "left_vendor_id",
        "right_vendor_id",
        "overlap_from",
        "overlap_to",
    ])


def unexplained_price_jumps(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    threshold: float = 0.50,
    action_tolerance_days: int = 3,
) -> pd.DataFrame:
    """Preserve every large raw-close move and classify nearby actions."""
    required_prices = {"ticker", "date", "closeunadj"}
    required_actions = {"ticker", "date", "action"}
    if not required_prices <= set(prices.columns):
        raise SharadarError(
            f"SEP jump audit missing columns: {sorted(required_prices - set(prices.columns))}"
        )
    if not required_actions <= set(actions.columns):
        raise SharadarError(
            f"ACTIONS jump audit missing columns: "
            f"{sorted(required_actions - set(actions.columns))}"
        )
    frame = prices[["ticker", "date", "closeunadj"]].copy()
    frame["ticker"] = frame["ticker"].map(_ticker_norm)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["ticker", "date"])
    frame["prior_closeunadj"] = frame.groupby("ticker")["closeunadj"].shift()
    frame["raw_return"] = frame["closeunadj"] / frame["prior_closeunadj"] - 1.0
    jumps = frame[frame["raw_return"].abs() > threshold].copy()

    action_frame = actions[["ticker", "date", "action"]].copy()
    action_frame["ticker"] = action_frame["ticker"].map(_ticker_norm)
    action_frame["date"] = pd.to_datetime(action_frame["date"], errors="raise")
    tolerance = pd.Timedelta(days=action_tolerance_days)
    compatible_keywords = (
        "split", "dividend", "distribution", "spinoff", "spin-off",
        "merger", "acquisition", "delist", "bankruptcy", "rights",
    )
    nearby_actions = []
    compatible = []
    for row in jumps.itertuples(index=False):
        candidates = action_frame[
            (action_frame["ticker"] == row.ticker)
            & (action_frame["date"] >= row.date - tolerance)
            & (action_frame["date"] <= row.date + tolerance)
        ]
        values = sorted({str(value).strip().lower() for value in candidates["action"]})
        nearby_actions.append("|".join(values))
        compatible.append(any(
            keyword in value for value in values for keyword in compatible_keywords
        ))
    jumps["nearby_actions"] = nearby_actions
    jumps["compatible_action"] = compatible
    return jumps[[
        "ticker",
        "date",
        "prior_closeunadj",
        "closeunadj",
        "raw_return",
        "nearby_actions",
        "compatible_action",
    ]].reset_index(drop=True)


def verify_price_jumps(
    jumps: pd.DataFrame,
    actions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    action_tolerance_days: int = 3,
    split_rel_tolerance: float = 0.15,
    dividend_rel_tolerance: float = 0.15,
) -> pd.DataFrame:
    """Mathematically verify large raw-close moves against vendor records.

    A jump is verified only when (a) a split-like ACTIONS row within
    tolerance carries a positive ratio value matching the closeunadj ratio,
    or (b) the SEP dividends column on or near the jump date implies the
    observed drop as one large special distribution. Ordinary dividends can
    never satisfy the distribution identity for a >50% move, and an ACTIONS
    keyword alone verifies nothing.
    """
    out = jumps.copy()
    action_frame = actions.copy()
    action_frame["ticker"] = action_frame["ticker"].map(_ticker_norm)
    action_frame["date"] = pd.to_datetime(action_frame["date"], errors="raise")
    action_frame["action_norm"] = (
        action_frame["action"].astype(str).str.strip().str.lower()
    )
    has_split_value = "value" in action_frame.columns
    if has_split_value:
        action_frame["value"] = pd.to_numeric(action_frame["value"], errors="coerce")
    dividends = None
    if "dividends" in prices.columns:
        dividends = prices[["ticker", "date", "dividends"]].copy()
        dividends["ticker"] = dividends["ticker"].map(_ticker_norm)
        dividends["date"] = pd.to_datetime(dividends["date"], errors="raise")
        dividends["dividends"] = pd.to_numeric(dividends["dividends"], errors="coerce")
        dividends = dividends[dividends["dividends"] > 0]
    tolerance = pd.Timedelta(days=action_tolerance_days)
    methods = []
    details = []
    for row in out.itertuples(index=False):
        method = ""
        detail = ""
        ratio = row.closeunadj / row.prior_closeunadj
        if has_split_value:
            splits = action_frame[
                (action_frame["ticker"] == row.ticker)
                & (action_frame["date"] >= row.date - tolerance)
                & (action_frame["date"] <= row.date + tolerance)
                & action_frame["action_norm"].str.contains("split")
                & action_frame["value"].notna()
                & (action_frame["value"] > 0)
            ]
            for value in splits["value"]:
                if abs(ratio * float(value) - 1.0) <= split_rel_tolerance:
                    method = "split_math"
                    detail = (
                        f"split value {float(value)} matches raw ratio {ratio:.6f}"
                    )
                    break
        if not method and dividends is not None and row.raw_return < 0:
            nearby = dividends[
                (dividends["ticker"] == row.ticker)
                & (dividends["date"] >= row.date - tolerance)
                & (dividends["date"] <= row.date + tolerance)
            ]
            for amount in nearby["dividends"]:
                implied = -float(amount) / row.prior_closeunadj
                if (
                    abs(row.raw_return - implied)
                    <= dividend_rel_tolerance * abs(row.raw_return)
                ):
                    method = "distribution_math"
                    detail = (
                        f"dividend {float(amount)} implies return {implied:.6f}"
                    )
                    break
        methods.append(method)
        details.append(detail)
    out["verified_method"] = methods
    out["verification_detail"] = details
    out["math_verified"] = [bool(method) for method in methods]
    return out


def price_jump_review(
    jumps: pd.DataFrame,
    approvals_path: Path | None,
    *,
    snapshot: str,
    tickers_sha256: str,
) -> pd.DataFrame:
    """Adjudicate verified jumps: math auto-passes, the rest need approvals."""
    approvals = pd.DataFrame(columns=sorted(REQUIRED_JUMP_APPROVAL_COLUMNS))
    if approvals_path is not None:
        approvals = pd.read_csv(approvals_path, dtype=str, keep_default_na=False)
        missing = REQUIRED_JUMP_APPROVAL_COLUMNS - set(approvals.columns)
        if missing:
            raise SharadarError(
                f"price-jump approvals missing columns: {sorted(missing)}"
            )
        if approvals["jump_id"].duplicated().any():
            raise SharadarError("price-jump approvals contain duplicate jump_id values")
    approval_map = (
        approvals.set_index("jump_id").to_dict("index") if len(approvals) else {}
    )
    out = jumps.copy()
    jump_ids = []
    hashes = []
    statuses = []
    evidence_values = []
    for row in out.itertuples(index=False):
        observed = {
            "ticker": str(row.ticker),
            "date": str(pd.Timestamp(row.date).date()),
            "prior_closeunadj": float(row.prior_closeunadj),
            "closeunadj": float(row.closeunadj),
            "raw_return": float(row.raw_return),
        }
        serialized = json.dumps(observed, sort_keys=True, separators=(",", ":"))
        observed_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        jump_id = f"{observed['ticker']}:{observed['date']}"
        approval = approval_map.get(jump_id, {})
        approved = (
            str(approval.get("status", "")).lower() == "approved"
            and _nonblank(approval.get("evidence"))
            and str(approval.get("snapshot", "")) == snapshot
            and str(approval.get("observed_sha256", "")) == observed_sha
            and str(approval.get("tickers_sha256", "")) == tickers_sha256
        )
        if row.math_verified:
            status = "auto_verified"
        elif approved:
            status = "approved"
        else:
            status = "review_required"
        jump_ids.append(jump_id)
        hashes.append(observed_sha)
        statuses.append(status)
        evidence_values.append(
            "" if pd.isna(approval.get("evidence"))
            else str(approval.get("evidence", ""))
        )
    out["jump_id"] = jump_ids
    out["observed_sha256"] = hashes
    out["manual_status"] = statuses
    out["evidence"] = evidence_values
    return out


def primary_key_audit(
    snapshot_dir: Path,
    *,
    duplicate_rows_max: int,
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    issues = []
    for table_dir in sorted(path for path in snapshot_dir.iterdir() if path.is_dir()):
        table = table_dir.name.upper()
        if table not in TABLE_SPECS:
            continue
        spec = TABLE_SPECS[table]
        nonnullable_key = [
            column for column in spec.primary_key if column not in spec.nullable_primary_key
        ]
        null_rows = 0
        row_count = 0
        parquet_files = [
            pq.ParquetFile(path) for path in sorted(table_dir.glob("*.parquet"))
        ]
        hashes = np.empty(
            sum(parquet.metadata.num_rows for parquet in parquet_files),
            dtype=np.uint64,
        )
        for parquet in parquet_files:
            for batch in parquet.iter_batches(
                batch_size=250_000,
                columns=list(spec.primary_key),
            ):
                frame = batch.to_pandas()
                batch_start = row_count
                row_count += len(frame)
                null_rows += int(frame[nonnullable_key].isna().any(axis=1).sum())
                hashes[batch_start:row_count] = pd.util.hash_pandas_object(
                    frame[list(spec.primary_key)], index=False, categorize=True
                ).to_numpy(dtype=np.uint64, copy=False)
        if len(hashes):
            hashes.sort()
            starts = np.r_[0, np.flatnonzero(hashes[1:] != hashes[:-1]) + 1]
            counts = np.diff(np.r_[starts, len(hashes)])
            duplicate_rows = int(counts[counts > 1].sum())
        else:
            duplicate_rows = 0
        rows.append({
            "table": table,
            "rows": row_count,
            "null_primary_key_rows": null_rows,
            "duplicate_primary_key_rows": duplicate_rows,
        })
        if null_rows:
            issues.append({
                "gate": "primary_key",
                "severity": "high",
                "subject": table,
                "detail": f"{null_rows} rows have null nonnullable primary-key fields",
            })
        if duplicate_rows > duplicate_rows_max:
            issues.append({
                "gate": "primary_key",
                "severity": "high",
                "subject": table,
                "detail": f"duplicate_rows={duplicate_rows} > {duplicate_rows_max}",
            })
    return pd.DataFrame(rows), issues


def _write_report(path: Path, run: dict, issues: pd.DataFrame) -> None:
    lines = [
        "# Sharadar blind QA",
        "",
        f"Status: **{run['status']}**",
        "",
        f"Snapshot: `{run['snapshot']}`",
        f"Git commit: `{run['git_commit']}` "
        f"(dirty: {run['dirty']}, {run['dirty_path_count']} paths)",
        f"Policy hash: `{run['policy_hash']}`",
        "",
        "## Gates",
        "",
        f"- Departed target: {run['departed_target']['covered']}/"
        f"{run['departed_target']['target']} ({run['departed_target']['coverage']:.3%})",
        f"- Member-day price coverage: {run['price_coverage']['coverage']:.3%}",
        f"- Minimum annual price coverage: {run['price_coverage']['min_year_coverage']:.3%}",
        f"- SEP rows quarantined by identity resolution: "
        f"{run['sep_row_resolution']['quarantined_rows']}",
        f"- Minimum month-end membership Jaccard: "
        f"{run['membership']['min_jaccard']:.3%}",
        f"- Known identity cases approved: {run['known_cases']['approved']}/"
        f"{run['known_cases']['total']}",
        f"- Listing/issuer registry coverage: "
        f"{run['identity']['registry']['coverage']:.3%}",
        f"- Raw-close jumps over 50%: {run['price_jumps']['total_large_moves']} "
        f"({run['price_jumps']['math_verified']} math-verified, "
        f"{run['price_jumps']['approved']} approved, "
        f"{run['price_jumps']['review_required']} unexplained)",
        "",
        "## Issues",
        "",
    ]
    if issues.empty:
        lines.append("None.")
    else:
        for row in issues.itertuples(index=False):
            lines.append(f"- [{row.severity.upper()}] {row.gate} / {row.subject}: {row.detail}")
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _path_has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _audit_snapshot_unlocked(
    *,
    snapshot: str,
    staged_root: Path = STAGED_ROOT,
    raw_root: Path = RAW_ROOT,
    membership_csv: Path = MEMBERSHIP_CSV,
    panel_path: Path = CACHE_DIR / "market_data.parquet",
    thresholds_path: Path = DEFAULT_THRESHOLDS,
    cases_path: Path = DEFAULT_CASES,
    overrides_path: Path | None = None,
    approvals_path: Path | None = None,
    membership_approvals_path: Path | None = None,
    identity_registry_path: Path | None = None,
    jump_approvals_path: Path | None = None,
    output_dir: Path | None = None,
    start: str = "2011-04-06",
    end: str | None = None,
    enforce_license: bool = True,
) -> dict:
    validate_snapshot_id(snapshot)
    snapshot_dir = staged_root / snapshot
    manifest_path = raw_root / snapshot / "manifest.json"
    if not manifest_path.exists():
        raise SharadarError(f"missing raw manifest for snapshot {snapshot}")
    manifest = verify_snapshot_manifest(
        manifest_path,
        raw_snapshot=raw_root / snapshot,
        staged_snapshot=snapshot_dir,
        enforce_license=enforce_license,
    )
    input_paths = {
        "membership": membership_csv,
        "panel": panel_path,
        "thresholds": thresholds_path,
        "cases": cases_path,
        "overrides": overrides_path,
        "known_case_approvals": approvals_path,
        "membership_approvals": membership_approvals_path,
        "identity_registry": identity_registry_path,
        "jump_approvals": jump_approvals_path,
    }
    input_sha256 = {
        key: (sha256_file(path) if path is not None else None)
        for key, path in input_paths.items()
    }
    policy_hash = hashlib.sha256(json.dumps(
        {key: input_sha256[key] for key in POLICY_INPUT_KEYS},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    thresholds = validate_thresholds(_read_json(thresholds_path))
    cases_config = validate_cases_config(_read_json(cases_path))

    reference_snapshots = load_reference_snapshots_raw(
        membership_csv,
        xs_band=(thresholds["cross_section_min"], thresholds["cross_section_max"]),
    )
    reference_end = reference_snapshots[-1][0]
    end_ts = min(pd.Timestamp(end), reference_end) if end else reference_end
    start_ts = pd.Timestamp(start)
    intervals = membership_intervals(reference_snapshots, start=start_ts, end=end_ts)

    tickers = prepare_vendor_tickers(read_staged_table(snapshot_dir, "TICKERS"))
    tickers_sha = staged_table_sha256(snapshot_dir, "TICKERS")
    membership_binding = json.dumps({
        "source_sha256": sha256_file(membership_csv),
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
    }, sort_keys=True, separators=(",", ":"))
    membership_sha = hashlib.sha256(membership_binding.encode("utf-8")).hexdigest()
    overrides = load_identity_overrides(
        overrides_path,
        snapshot=snapshot,
        tickers_sha256=tickers_sha,
        membership_sha256=membership_sha,
    )
    crosswalk = resolve_identity_intervals(intervals, tickers, overrides)
    prices = read_staged_table(snapshot_dir, "SEP")
    panel = pd.read_parquet(panel_path, columns=["ticker", "date"])
    panel["date"] = pd.to_datetime(panel["date"])
    calendar = pd.DatetimeIndex(
        panel.loc[
            (panel["date"] >= start_ts) & (panel["date"] <= end_ts), "date"
        ]
        .drop_duplicates()
        .sort_values()
    )
    if calendar.empty:
        raise SharadarError("independent panel contains no dates inside the audit window")
    calendar_stats = calendar_quality(calendar, start=start_ts, end=end_ts)
    calendar_weekday_coverage = calendar_stats["weekday_coverage"]
    calendar_start_gap = calendar_stats["start_gap_days"]
    calendar_end_gap = calendar_stats["end_gap_days"]
    calendar_min_month_coverage = calendar_stats["min_month_weekday_coverage"]
    calendar_internal_gap = calendar_stats["max_internal_gap_days"]
    panel_identities = resolve_panel_identities(
        panel,
        tickers,
        as_of=end_ts,
        recency_days=7,
    )
    panel_vendor_ids = set(
        panel_identities.loc[
            (panel_identities["match_status"] == "resolved")
            & panel_identities["current_at_audit_end"],
            "vendor_id",
        ].dropna()
    )
    sp500 = read_staged_table(snapshot_dir, "SP500")
    membership_vendor = membership_reconciliation(
        sp500,
        reference_snapshots,
        tickers,
        start=start_ts,
        end=end_ts,
    )
    relevant_vendor_ids = set(crosswalk["vendor_id"].dropna().astype(str)) | {
        str(value) for value in panel_vendor_ids
    }
    sp500_symbols = {_ticker_norm(value) for value in sp500["ticker"].dropna()}
    relevant_vendor_ids.update(
        tickers.loc[tickers["ticker_norm"].isin(sp500_symbols), "vendor_id"]
        .dropna().astype(str)
    )
    for serialized in membership_vendor["vendor_only_ids"].dropna():
        relevant_vendor_ids.update(
            value for value in str(serialized).split("|") if value
        )
    identity_registry, registry_summary = identity_registry_report(
        identity_registry_path,
        snapshot=snapshot,
        tickers_sha256=tickers_sha,
        relevant_vendor_ids=relevant_vendor_ids,
    )
    approved_registry = identity_registry[
        identity_registry["bound_and_approved"]
    ][["vendor_id", "listing_id", "issuer_id"]]
    identity_map: dict[str, str] = {}
    if len(approved_registry):
        listing_map = approved_registry.set_index("vendor_id")["listing_id"]
        issuer_map = approved_registry.set_index("vendor_id")["issuer_id"]
        identity_map = {str(key): str(value) for key, value in listing_map.items()}
        crosswalk["listing_id"] = crosswalk["vendor_id"].map(listing_map)
        crosswalk["issuer_id"] = crosswalk["vendor_id"].map(issuer_map)
        panel_identities["listing_id"] = panel_identities["vendor_id"].map(listing_map)
        panel_identities["issuer_id"] = panel_identities["vendor_id"].map(issuer_map)
    else:
        panel_identities["listing_id"] = None
        panel_identities["issuer_id"] = None
    sep_resolution, sep_quarantine = resolve_sep_rows(
        prices, tickers, identity_map=identity_map
    )
    interval_coverage, by_year, price_summary = price_coverage(
        crosswalk,
        prices,
        calendar,
        row_resolution=sep_resolution,
        identity_map=identity_map,
    )
    numeric_prices = prices[list(PRICE_NUMERIC_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_price_rows = int(
        (~(numeric_prices.notna().all(axis=1)
           & np.isfinite(numeric_prices).all(axis=1))).sum()
    )
    target_crosswalk = crosswalk.copy()
    target_crosswalk["vendor_id"] = target_crosswalk.apply(
        lambda row: (
            row["listing_id"]
            if _nonblank(row["listing_id"])
            else (
                f"unverified:{row['vendor_id']}"
                if _nonblank(row["vendor_id"]) else None
            )
        ),
        axis=1,
    )
    panel_identity_ids = {
        identity_map.get(str(value), f"unverified:{value}")
        for value in panel_vendor_ids
    }
    departed_rows, departed_summary = departed_target_coverage(
        target_crosswalk,
        interval_coverage,
        panel_identity_ids,
        interval_coverage_min=thresholds["departed_interval_price_coverage_min"],
    )
    membership = membership_reconciliation(
        sp500,
        reference_snapshots,
        tickers,
        start=start_ts,
        end=end_ts,
        identity_map=identity_map,
    )
    membership_daily = membership_reconciliation(
        sp500,
        reference_snapshots,
        tickers,
        start=start_ts,
        end=end_ts,
        comparison_dates=calendar,
        identity_map=identity_map,
    )
    membership_differences = membership_difference_report(
        membership_daily,
        membership_approvals_path,
        snapshot=snapshot,
        tickers_sha256=tickers_sha,
        membership_sha256=membership_sha,
    )
    known = known_case_report(
        tickers,
        cases_config,
        approvals_path,
        snapshot=snapshot,
        tickers_sha256=tickers_sha,
        cases_sha256=sha256_file(cases_path),
    )
    primary_keys, issues_list = primary_key_audit(
        snapshot_dir,
        duplicate_rows_max=thresholds["unresolved_primary_key_duplicates_max"],
    )
    alias_overlaps = alias_window_audit(tickers)
    actions = read_staged_table(snapshot_dir, "ACTIONS")
    price_jump_audit = price_jump_review(
        verify_price_jumps(unexplained_price_jumps(prices, actions), actions, prices),
        jump_approvals_path,
        snapshot=snapshot,
        tickers_sha256=tickers_sha,
    )
    price_jumps = price_jump_audit[
        price_jump_audit["manual_status"] == "review_required"
    ].copy()

    ambiguous = int(crosswalk["match_status"].isin({"ambiguous", "ambiguous_override"}).sum())
    if ambiguous > thresholds["unresolved_identity_ambiguities_max"]:
        issues_list.append({
            "gate": "identity",
            "severity": "high",
            "subject": "crosswalk",
            "detail": f"{ambiguous} unresolved ambiguous intervals",
        })
    unmatched = int((crosswalk["match_status"] == "unmatched").sum())
    if unmatched:
        issues_list.append({
            "gate": "identity_quarantine",
            "severity": "review",
            "subject": "crosswalk",
            "detail": f"{unmatched} unmatched intervals require mapping or explicit adjudication",
        })
    invalid_override_bindings = int(
        (~overrides["binding_valid"]).sum()
    ) if "binding_valid" in overrides else 0
    if invalid_override_bindings:
        issues_list.append({
            "gate": "identity_override_binding",
            "severity": "review",
            "subject": "manual_overrides",
            "detail": f"{invalid_override_bindings} overrides are stale or unbound",
        })
    unresolved_panel = int((
        panel_identities["current_at_audit_end"]
        & (panel_identities["match_status"] != "resolved")
    ).sum())
    if unresolved_panel:
        issues_list.append({
            "gate": "panel_identity_resolution",
            "severity": "high",
            "subject": "survivor_panel",
            "detail": f"{unresolved_panel} panel series do not resolve to one vendor identity",
        })
    if registry_summary["coverage"] < thresholds["identity_registry_coverage_min"]:
        issues_list.append({
            "gate": "listing_issuer_registry",
            "severity": "review",
            "subject": "identity_semantics",
            "detail": f"{registry_summary['coverage']:.3%} < "
                      f"{thresholds['identity_registry_coverage_min']:.3%} "
                      "snapshot-bound listing/issuer coverage",
        })
    if len(alias_overlaps):
        issues_list.append({
            "gate": "alias_window_overlap",
            "severity": "high",
            "subject": "TICKERS",
            "detail": f"{len(alias_overlaps)} ticker windows overlap across vendor identities",
        })
    if len(price_jumps):
        issues_list.append({
            "gate": "price_jump_review",
            "severity": "high",
            "subject": "SEP",
            "detail": f"{len(price_jumps)} raw-close moves exceed 50% without "
                      "mathematical verification or a snapshot-bound approval",
        })
    if invalid_price_rows:
        issues_list.append({
            "gate": "price_numeric_validity",
            "severity": "high",
            "subject": "SEP",
            "detail": f"{invalid_price_rows} rows have null or non-finite price fields",
        })
    if len(sep_quarantine) > thresholds["sep_quarantined_rows_max"]:
        issues_list.append({
            "gate": "sep_row_quarantine",
            "severity": "high",
            "subject": "SEP",
            "detail": f"{len(sep_quarantine)} rows resolve to zero or multiple "
                      f"listings > {thresholds['sep_quarantined_rows_max']}",
        })
    if calendar_weekday_coverage < thresholds["calendar_weekday_coverage_min"]:
        issues_list.append({
            "gate": "independent_calendar",
            "severity": "high",
            "subject": "market_panel",
            "detail": f"weekday density {calendar_weekday_coverage:.3%} < "
                      f"{thresholds['calendar_weekday_coverage_min']:.3%}",
        })
    if calendar_min_month_coverage < thresholds["calendar_month_weekday_coverage_min"]:
        issues_list.append({
            "gate": "independent_calendar",
            "severity": "high",
            "subject": "market_panel_worst_month",
            "detail": f"weekday density {calendar_min_month_coverage:.3%} < "
                      f"{thresholds['calendar_month_weekday_coverage_min']:.3%}",
        })
    endpoint_gap = max(calendar_start_gap, calendar_end_gap)
    if endpoint_gap > thresholds["calendar_endpoint_gap_days_max"]:
        issues_list.append({
            "gate": "independent_calendar",
            "severity": "high",
            "subject": "market_panel",
            "detail": f"endpoint gap {endpoint_gap}d > "
                      f"{thresholds['calendar_endpoint_gap_days_max']}d",
        })
    if calendar_internal_gap > thresholds["calendar_internal_gap_days_max"]:
        issues_list.append({
            "gate": "independent_calendar",
            "severity": "high",
            "subject": "market_panel_internal_gap",
            "detail": f"internal gap {calendar_internal_gap}d > "
                      f"{thresholds['calendar_internal_gap_days_max']}d",
        })
    if len(calendar) != thresholds["calendar_expected_dates"]:
        issues_list.append({
            "gate": "calendar_expectation",
            "severity": "high",
            "subject": "market_panel",
            "detail": f"{len(calendar)} calendar dates != expected "
                      f"{thresholds['calendar_expected_dates']}",
        })
    if calendar_stats["sha256"] != thresholds["calendar_expected_sha256"]:
        issues_list.append({
            "gate": "calendar_expectation",
            "severity": "high",
            "subject": "market_panel",
            "detail": f"calendar sha256 {calendar_stats['sha256']} != expected "
                      f"{thresholds['calendar_expected_sha256']}",
        })
    if departed_summary["coverage"] < thresholds["departed_identity_coverage_min"]:
        issues_list.append({
            "gate": "departed_identity_coverage",
            "severity": "high",
            "subject": "historical_panel",
            "detail": f"{departed_summary['coverage']:.3%} < "
                      f"{thresholds['departed_identity_coverage_min']:.3%}",
        })
    if price_summary["coverage"] < thresholds["member_day_price_coverage_min"]:
        issues_list.append({
            "gate": "member_day_price_coverage",
            "severity": "high",
            "subject": "all_years",
            "detail": f"{price_summary['coverage']:.3%} < "
                      f"{thresholds['member_day_price_coverage_min']:.3%}",
        })
    if price_summary["min_year_coverage"] < thresholds["member_day_price_coverage_by_year_min"]:
        issues_list.append({
            "gate": "member_day_price_coverage",
            "severity": "high",
            "subject": "worst_year",
            "detail": f"{price_summary['min_year_coverage']:.3%} < "
                      f"{thresholds['member_day_price_coverage_by_year_min']:.3%}",
        })
    min_jaccard = float(membership["jaccard"].min()) if len(membership) else 0.0
    unresolved_membership = int(
        (membership["unresolved_reference"] + membership["unresolved_vendor"]).max()
    ) if len(membership) else 0
    if unresolved_membership:
        issues_list.append({
            "gate": "membership_identity_resolution",
            "severity": "high",
            "subject": "month_end_membership",
            "detail": f"up to {unresolved_membership} unresolved symbols in a month",
        })
    daily_unresolved = int(
        (membership_daily["unresolved_reference"]
         + membership_daily["unresolved_vendor"]).max()
    ) if len(membership_daily) else 0
    if daily_unresolved:
        issues_list.append({
            "gate": "membership_identity_resolution",
            "severity": "high",
            "subject": "daily_membership",
            "detail": f"up to {daily_unresolved} unresolved symbols on a trading day",
        })
    difference_approved = int(
        (membership_differences["manual_status"] == "approved").sum()
    )
    difference_rate = (
        difference_approved / len(membership_differences)
        if len(membership_differences) else 1.0
    )
    if difference_rate < thresholds["membership_difference_approval_rate_min"]:
        issues_list.append({
            "gate": "membership_difference_adjudication",
            "severity": "review",
            "subject": "persistent_differences",
            "detail": f"{difference_approved}/{len(membership_differences)} approved; "
                      f"required rate "
                      f"{thresholds['membership_difference_approval_rate_min']:.1%}",
        })
    if min_jaccard < thresholds["membership_month_end_jaccard_min"]:
        issues_list.append({
            "gate": "membership_jaccard",
            "severity": "high",
            "subject": "worst_month",
            "detail": f"{min_jaccard:.3%} < "
                      f"{thresholds['membership_month_end_jaccard_min']:.3%}",
        })
    xs_bad = membership[
        (membership["vendor_raw_count"] < thresholds["cross_section_min"])
        | (membership["vendor_raw_count"] > thresholds["cross_section_max"])
    ]
    if len(xs_bad):
        issues_list.append({
            "gate": "membership_cross_section",
            "severity": "high",
            "subject": "SP500",
            "detail": f"{len(xs_bad)} month-end rows outside "
                      f"[{thresholds['cross_section_min']}, {thresholds['cross_section_max']}]",
        })
    approved = int((known["manual_status"] == "approved").sum())
    approval_rate = approved / len(known) if len(known) else 0.0
    if approval_rate < thresholds["known_case_approval_rate_min"]:
        issues_list.append({
            "gate": "known_identity_cases",
            "severity": "review",
            "subject": "manual_adjudication",
            "detail": f"{approved}/{len(known)} approved with evidence; required rate "
                      f"{thresholds['known_case_approval_rate_min']:.1%}",
        })

    issues = pd.DataFrame(issues_list, columns=["gate", "severity", "subject", "detail"])
    hard_fail = bool((issues["severity"] == "high").any()) if len(issues) else False
    needs_review = bool((issues["severity"] == "review").any()) if len(issues) else False
    gate_status = "FAIL" if hard_fail else ("REVIEW" if needs_review else "PASS")
    canonical_policy = (
        enforce_license
        and end is None
        and str(pd.Timestamp(start).date()) == "2011-04-06"
        and thresholds_path.resolve() == DEFAULT_THRESHOLDS.resolve()
        and cases_path.resolve() == DEFAULT_CASES.resolve()
        and membership_csv.resolve() == MEMBERSHIP_CSV.resolve()
        and panel_path.resolve() == (CACHE_DIR / "market_data.parquet").resolve()
    )
    status = (
        gate_status
        if canonical_policy or gate_status != "PASS"
        else "NONCANONICAL"
    )
    git_commit = _git_commit()
    worktree = _git_worktree()
    run = {
        "schema_version": 1,
        "status": status,
        "gate_status": gate_status,
        "canonical_policy": canonical_policy,
        "policy_id": "vendor_qa_v1-full-window-license-enforced",
        "policy_hash": policy_hash,
        "input_sha256": input_sha256,
        "snapshot": snapshot,
        "git_commit": git_commit,
        "dirty": worktree["dirty"],
        "dirty_path_count": worktree["dirty_path_count"],
        "canonical_pass": (
            {
                "git_commit": git_commit,
                "dirty": worktree["dirty"],
                "policy_hash": policy_hash,
            }
            if status == "PASS" else None
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "license_expires": manifest["license_expires"],
        "reference_window": {"start": str(start_ts.date()), "end": str(end_ts.date())},
        "thresholds": thresholds,
        "dynamic_denominator": {
            "reference_intervals": len(intervals),
            "reference_tickers": int(intervals["reference_ticker"].nunique()),
            "current_panel_series": int(panel_identities["current_at_audit_end"].sum()),
            "resolved_panel_vendor_ids": len(panel_vendor_ids),
        },
        "departed_target": departed_summary,
        "price_coverage": {
            **price_summary,
            "calendar_source": "union of independent market-panel dates",
            "calendar_dates": len(calendar),
            "calendar_sha256": calendar_stats["sha256"],
            "calendar_weekday_coverage": calendar_weekday_coverage,
            "calendar_min_month_weekday_coverage": calendar_min_month_coverage,
            "calendar_start_gap_days": calendar_start_gap,
            "calendar_end_gap_days": calendar_end_gap,
            "calendar_max_internal_gap_days": calendar_internal_gap,
            "invalid_numeric_rows": invalid_price_rows,
        },
        "sep_row_resolution": {
            "rows": int(len(prices)),
            "resolved_rows": int(len(sep_resolution)),
            "quarantined_rows": int(len(sep_quarantine)),
            "quarantined_zero_listing_rows": int(
                (sep_quarantine["quarantine_reason"] == "zero_listings").sum()
            ),
            "quarantined_multiple_listing_rows": int(
                (sep_quarantine["quarantine_reason"] == "multiple_listings").sum()
            ),
        },
        "membership": {
            "months": len(membership),
            "daily_checkpoints": len(membership_daily),
            "min_jaccard": min_jaccard,
            "months_below_threshold": int(
                (membership["jaccard"] < thresholds["membership_month_end_jaccard_min"]).sum()
            ),
            "persistent_differences": len(membership_differences),
            "persistent_differences_approved": difference_approved,
        },
        "identity": {
            "resolved_intervals": int((crosswalk["match_status"] == "resolved").sum()),
            "unmatched_intervals": unmatched,
            "ambiguous_intervals": ambiguous,
            "vendor_id_semantics": registry_summary["status"],
            "registry": registry_summary,
            "alias_window_overlaps": len(alias_overlaps),
        },
        "known_cases": {"approved": approved, "total": len(known)},
        "price_jumps": {
            "total_large_moves": int(len(price_jump_audit)),
            "math_verified": int(
                (price_jump_audit["manual_status"] == "auto_verified").sum()
            ),
            "approved": int(
                (price_jump_audit["manual_status"] == "approved").sum()
            ),
            "review_required": int(len(price_jumps)),
        },
        "unexplained_price_jumps": len(price_jumps),
        "primary_keys": primary_keys.to_dict("records"),
        "issue_count": len(issues),
    }

    recomputed_sha256 = {
        key: (sha256_file(path) if path is not None else None)
        for key, path in input_paths.items()
    }
    if recomputed_sha256 != input_sha256:
        changed = sorted(
            key for key in input_paths
            if recomputed_sha256[key] != input_sha256[key]
        )
        raise SharadarError(
            f"QA inputs changed during the audit: {changed}; no report written"
        )

    canonical_output = Path(manifest["storage_roots"]["qa"]) / snapshot
    output = output_dir or canonical_output
    output_resolved = output.resolve()
    raw_snapshot = (raw_root / snapshot).resolve()
    staged_snapshot = snapshot_dir.resolve()
    if (
        output_resolved == raw_snapshot
        or output_resolved.is_relative_to(raw_snapshot)
        or output_resolved == staged_snapshot
        or output_resolved.is_relative_to(staged_snapshot)
    ):
        raise SharadarError("QA output must not be inside raw or staged snapshot storage")
    if enforce_license and output_resolved != canonical_output.resolve():
        raise SharadarError("licensed QA output must use the manifest-tracked QA root")
    if _path_has_symlink_component(output):
        raise SharadarError("QA output path must not contain symlinks")
    if output.exists() and any(path.is_symlink() for path in output.rglob("*")):
        raise SharadarError("QA output directory contains symlinked entries")
    output.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(crosswalk.merge(interval_coverage, on=[
        "interval_id", "reference_ticker", "vendor_ticker", "vendor_id", "match_status"
    ], how="left", validate="one_to_one"), output / "identity_crosswalk.parquet")
    for frame, name in (
        (departed_rows, "departed_target.parquet"),
        (sep_quarantine, "sep_quarantine.parquet"),
        (panel_identities, "panel_identity_crosswalk.parquet"),
        (identity_registry, "identity_registry_review.parquet"),
        (alias_overlaps, "alias_window_overlaps.parquet"),
        (price_jumps, "unexplained_price_jumps.parquet"),
        (price_jump_audit, "all_large_price_jumps.parquet"),
        (by_year, "coverage_by_year.parquet"),
        (membership, "membership_by_month.parquet"),
        (membership_daily, "membership_by_day.parquet"),
        (membership_differences, "membership_differences.parquet"),
        (known, "known_cases.parquet"),
        (issues, "issues.parquet"),
    ):
        _write_parquet_atomic(frame, output / name)
    run_path = output / "run.json"
    run_tmp = run_path.with_suffix(".json.part")
    run_tmp.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    run_tmp.replace(run_path)
    _write_report(output / "report.md", run, issues)
    verify_snapshot_manifest(
        manifest_path,
        raw_snapshot=raw_root / snapshot,
        staged_snapshot=snapshot_dir,
        enforce_license=enforce_license,
    )
    return run


def audit_snapshot(
    *,
    snapshot: str,
    staged_root: Path = STAGED_ROOT,
    raw_root: Path = RAW_ROOT,
    membership_csv: Path = MEMBERSHIP_CSV,
    panel_path: Path = CACHE_DIR / "market_data.parquet",
    thresholds_path: Path = DEFAULT_THRESHOLDS,
    cases_path: Path = DEFAULT_CASES,
    overrides_path: Path | None = None,
    approvals_path: Path | None = None,
    membership_approvals_path: Path | None = None,
    identity_registry_path: Path | None = None,
    jump_approvals_path: Path | None = None,
    output_dir: Path | None = None,
    start: str = "2011-04-06",
    end: str | None = None,
    enforce_license: bool = True,
) -> dict:
    """Run one QA audit while fetch and purge are excluded for the snapshot."""
    qa_root = Path(output_dir).parent if output_dir is not None else Path(
        DATA_DIR / "qa" / "sharadar"
    )
    with snapshot_lock(
        raw_root,
        snapshot,
        additional_roots=(staged_root, qa_root),
    ):
        return _audit_snapshot_unlocked(
            snapshot=snapshot,
            staged_root=staged_root,
            raw_root=raw_root,
            membership_csv=membership_csv,
            panel_path=panel_path,
            thresholds_path=thresholds_path,
            cases_path=cases_path,
            overrides_path=overrides_path,
            approvals_path=approvals_path,
            membership_approvals_path=membership_approvals_path,
            identity_registry_path=identity_registry_path,
            jump_approvals_path=jump_approvals_path,
            output_dir=output_dir,
            start=start,
            end=end,
            enforce_license=enforce_license,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blind QA on a staged Sharadar snapshot")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--membership", default=str(MEMBERSHIP_CSV))
    parser.add_argument("--panel", default=str(CACHE_DIR / "market_data.parquet"))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--overrides")
    parser.add_argument("--approvals")
    parser.add_argument("--membership-approvals")
    parser.add_argument("--identity-registry")
    parser.add_argument("--jump-approvals")
    parser.add_argument("--output")
    parser.add_argument("--start", default="2011-04-06")
    parser.add_argument("--end")
    args = parser.parse_args()
    run = audit_snapshot(
        snapshot=args.snapshot,
        membership_csv=Path(args.membership),
        panel_path=Path(args.panel),
        thresholds_path=Path(args.thresholds),
        cases_path=Path(args.cases),
        overrides_path=Path(args.overrides) if args.overrides else None,
        approvals_path=Path(args.approvals) if args.approvals else None,
        membership_approvals_path=(
            Path(args.membership_approvals) if args.membership_approvals else None
        ),
        identity_registry_path=(
            Path(args.identity_registry) if args.identity_registry else None
        ),
        jump_approvals_path=(
            Path(args.jump_approvals) if args.jump_approvals else None
        ),
        output_dir=Path(args.output) if args.output else None,
        start=args.start,
        end=args.end,
    )
    print(json.dumps(run, indent=2, sort_keys=True))
    raise SystemExit(0 if run["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
