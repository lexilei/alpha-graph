"""Synthetic tests for staged Sharadar ingestion and blind identity QA."""

from __future__ import annotations

import json
import hashlib
import fcntl
from datetime import date

import pandas as pd
import pytest

from alpha_graph.data.sharadar import (
    DownloadPart,
    NasdaqBulkClient,
    SharadarAPIError,
    SharadarError,
    _NasdaqAPIRedirectHandler,
    _validate_and_stage,
    build_bulk_url,
    build_download_plan,
    fetch_snapshot,
    load_scope_tickers,
    plan_summary,
    purge_snapshot_data,
    sha256_file,
    validate_license_expiry,
    verify_snapshot_manifest,
    TABLE_SPECS,
)
from alpha_graph.data.sharadar_qa import (
    audit_snapshot,
    calendar_quality,
    departed_target_coverage,
    known_case_report,
    load_identity_overrides,
    load_reference_snapshots_raw,
    membership_intervals,
    membership_difference_report,
    membership_reconciliation,
    prepare_vendor_tickers,
    price_coverage,
    resolve_panel_identities,
    resolve_identity_intervals,
)


def _sep_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
        "open": [10.0, 10.5],
        "high": [11.0, 11.0],
        "low": [9.5, 10.0],
        "close": [10.5, 10.8],
        "volume": [1000, 1200],
        "closeadj": [9.8, 10.1],
        "closeunadj": [10.5, 10.8],
        "dividends": [0.0, 0.0],
        "lastupdated": pd.to_datetime(["2020-01-04", "2020-01-04"]),
        "future_vendor_column": [1, 2],
    })


def _ticker_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "table": ["SEP", "SEP", "SEP"],
        "permaticker": [101, 202, 303],
        "ticker": ["IR", "IR", "ZZZ"],
        "name": ["Old IR", "New IR", "Zed"],
        "firstpricedate": pd.to_datetime(["2000-01-01", "2020-03-01", "2010-01-01"]),
        "lastpricedate": pd.to_datetime(["2020-02-28", "2025-12-31", "2025-12-31"]),
    })


def _actions_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"]),
        "ticker": ["AAA"],
        "name": ["Alpha"],
        "action": ["dividend"],
        "contraname": [None],
        "contraticker": [None],
    })


def _write_complete_snapshot(raw_snapshot, staged_snapshot, tables):
    raw_snapshot.mkdir(parents=True, exist_ok=True)
    planned = []
    records = []
    for table, frame in tables.items():
        part = DownloadPart(table, 0, ())
        planned.append({
            "stem": part.stem,
            "table": table,
            "index": 0,
            "filters": [],
            "digest": part.digest,
        })
        raw_path = raw_snapshot / table / f"{part.stem}-file-000.parquet"
        staged_path = staged_snapshot / table / raw_path.name
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(raw_path, index=False)
        frame.to_parquet(staged_path, index=False)
        records.append({
            "stem": part.stem,
            "table": table,
            "index": 0,
            "filters": [],
            "requested_at_utc": "2020-01-01T00:00:00+00:00",
            "status": "SUCCEEDED",
            "empty_export": False,
            "schema": {"fixture": 1},
            "files": [{
                "raw_path": str(raw_path.relative_to(raw_snapshot)),
                "raw_size": raw_path.stat().st_size,
                "raw_sha256": sha256_file(raw_path),
                "staged_path": str(staged_path.relative_to(staged_snapshot)),
                "staged_sha256": sha256_file(staged_path),
                "rows": len(frame),
                "columns": list(frame.columns),
                "first_date": None,
                "last_date": None,
            }],
        })
    plan_payload = [
        {"table": item["table"], "index": item["index"], "filters": []}
        for item in planned
    ]
    manifest = {
        "schema_version": 2,
        "provider": "Sharadar via Nasdaq Data Link",
        "snapshot": raw_snapshot.name,
        "package": "SFA",
        "license_expires": "2099-01-01",
        "license_evidence_sha256": "fixture",
        "storage_roots": {
            "raw": str(raw_snapshot.parent.resolve()),
            "staged": str(staged_snapshot.parent.resolve()),
            "qa": str((raw_snapshot.parent.parent / "qa").resolve()),
        },
        "plan_sha256": hashlib.sha256(
            json.dumps(plan_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "planned_parts": planned,
        "status": "COMPLETE",
        "parts": records,
    }
    path = raw_snapshot / "manifest.json"
    path.write_text(json.dumps(manifest))
    (raw_snapshot / "license.json").write_text(json.dumps({
        "snapshot": raw_snapshot.name,
        "package": "SFA",
        "license_expires": "2099-01-01",
        "written_confirmation_sha256": "fixture",
    }))
    return path


def test_plan_is_deterministic_scoped_and_secret_free():
    tickers = [f"T{i:03d}" for i in range(83)]
    plan = build_download_plan(
        ["TICKERS", "SP500", "SEP"],
        tickers=tickers,
        start="2011-01-01",
        batch_size=40,
    )
    assert [p.table for p in plan].count("SEP") == 3
    assert [p.table for p in plan].count("TICKERS") == 1
    assert [p.table for p in plan].count("SP500") == 1
    sep = next(p for p in plan if p.table == "SEP")
    assert sum(key == "ticker.in[]" for key, _ in sep.filters) == 40
    assert "api_key" not in build_bulk_url(sep).lower()
    summary = plan_summary(plan)
    assert summary["requests_by_table"] == {"SEP": 3, "SP500": 1, "TICKERS": 1}
    assert summary == plan_summary(plan)


def test_plan_enforces_initial_per_table_limit():
    tickers = [f"T{i:04d}" for i in range(101)]
    with pytest.raises(ValueError, match="exceeds the initial"):
        build_download_plan(["SEP"], tickers=tickers, batch_size=4)


def test_sp500_plan_keeps_events_after_local_audit_end():
    part = build_download_plan(
        ["SP500"], start="2019-01-01", end="2020-12-31"
    )[0]
    assert ("date.gte", "2019-01-01") in part.filters
    assert not any(key == "date.lte" for key, _ in part.filters)


def test_scope_tickers_preserves_class_syntax_variants(tmp_path):
    membership = tmp_path / "membership.csv"
    pd.DataFrame({
        "date": ["2020-01-01"],
        "tickers": ["AAA,BRK.B"],
    }).to_csv(membership, index=False)
    tickers = load_scope_tickers(membership, start="2019-01-01", panel_path=None)
    assert "AAA" in tickers
    assert "BRK.B" in tickers
    assert "BRK-B" in tickers


def test_bulk_client_polls_and_uses_header_token_only(tmp_path):
    calls = []
    statuses = iter(["PENDING", "RUNNING", "SUCCEEDED"])

    def get_json(url, token, timeout):
        calls.append(("json", url, token, timeout))
        status = next(statuses)
        files = [{"url": "https://data.nasdaq.com/api/v1/bulkdownloads/file/x.parquet",
                  "size": 0}] if status == "SUCCEEDED" else []
        return {"bulk_download": {"status": status, "files": files}}

    def download(url, token, destination, timeout):
        calls.append(("file", url, token, timeout))
        _sep_frame().to_parquet(destination, index=False)
        return destination.stat().st_size

    client = NasdaqBulkClient(
        "top-secret",
        poll_seconds=0,
        json_getter=get_json,
        file_downloader=download,
        sleep=lambda _: None,
    )
    part = DownloadPart("SEP", 0, (("date.gte", "2020-01-01"),))
    job = client.wait_for_export(part)
    paths = client.download_files(job, tmp_path, part.stem)
    assert len(paths) == 1
    assert all(token == "top-secret" for _, _, token, _ in calls)
    assert all("top-secret" not in url for _, url, _, _ in calls)


def test_bulk_client_terminal_failure():
    def failed(url, token, timeout):
        return {"bulk_download": {"status": "FAILED", "files": [], "errors": "bad filter"}}

    client = NasdaqBulkClient("secret", json_getter=failed)
    with pytest.raises(SharadarAPIError, match="bad filter"):
        client.wait_for_export(DownloadPart("SEP", 0, ()))


def test_bulk_client_rejects_untrusted_file_url(tmp_path):
    called = False

    def download(url, token, destination, timeout):
        nonlocal called
        called = True
        return 0

    client = NasdaqBulkClient("secret", file_downloader=download)
    with pytest.raises(SharadarAPIError, match="approved Nasdaq HTTPS origin"):
        client.download_files(
            {"files": [{"url": "http://attacker.invalid/x.parquet"}]},
            tmp_path,
            "unsafe",
        )
    assert not called


def test_api_redirect_and_required_empty_exports_fail_closed():
    handler = _NasdaqAPIRedirectHandler()
    with pytest.raises(SharadarAPIError, match="approved Nasdaq HTTPS origin"):
        handler.redirect_request(
            None, None, 302, "redirect", {}, "https://attacker.invalid/job"
        )

    client = NasdaqBulkClient(
        "secret",
        json_getter=lambda url, token, timeout: {
            "bulk_download": {"status": "SUCCEEDED", "files": []}
        },
    )
    with pytest.raises(SharadarAPIError, match="unexpectedly empty"):
        client.wait_for_export(DownloadPart("SEP", 0, ()))


def test_download_retries_integrity_validation(tmp_path):
    calls = 0

    def download(url, token, destination, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            destination.write_bytes(b"not-parquet")
        else:
            _sep_frame().to_parquet(destination, index=False)
        return destination.stat().st_size

    client = NasdaqBulkClient(
        "secret",
        file_downloader=download,
        max_retries=1,
        retry_base_seconds=0,
        sleep=lambda _: None,
    )
    paths = client.download_files({"files": [{
        "url": "https://data.nasdaq.com/api/v1/bulkdownloads/file/x.parquet",
        "size": 0,
    }]}, tmp_path, "retry")
    assert calls == 2
    assert len(paths) == 1


def test_stage_preserves_price_bases_and_allows_extra_columns(tmp_path):
    raw = tmp_path / "raw.parquet"
    staged = tmp_path / "staged.parquet"
    _sep_frame().to_parquet(raw, index=False)
    result = _validate_and_stage(raw, staged, TABLE_SPECS["SEP"])
    out = pd.read_parquet(staged)
    assert result["rows"] == 2
    assert {"close", "closeadj", "closeunadj", "future_vendor_column"} <= set(out.columns)

    duplicate = pd.concat([_sep_frame(), _sep_frame().iloc[[0]]], ignore_index=True)
    duplicate.to_parquet(raw, index=False)
    with pytest.raises(SharadarError, match="duplicate primary keys"):
        _validate_and_stage(raw, staged, TABLE_SPECS["SEP"])

    _sep_frame().drop(columns="dividends").to_parquet(raw, index=False)
    with pytest.raises(SharadarError, match="dividends"):
        _validate_and_stage(raw, staged, TABLE_SPECS["SEP"])


def test_actions_allows_null_counterparty_fields(tmp_path):
    raw = tmp_path / "actions.parquet"
    staged = tmp_path / "staged.parquet"
    _actions_frame().to_parquet(raw, index=False)
    result = _validate_and_stage(raw, staged, TABLE_SPECS["ACTIONS"])
    assert result["rows"] == 1


def test_fetch_snapshot_is_resumable_and_manifest_has_no_secret(tmp_path):
    raw_root = tmp_path / "raw"
    staged_root = tmp_path / "staged"
    part = DownloadPart("SEP", 0, (("ticker.in[]", "AAA"),))
    license_evidence = tmp_path / "license-confirmation.txt"
    license_evidence.write_text("written vendor confirmation")
    calls = {"jobs": 0, "files": 0}

    def get_json(url, token, timeout):
        calls["jobs"] += 1
        return {"bulk_download": {
            "status": "SUCCEEDED",
            "files": [{"url": "https://data.nasdaq.com/api/v1/bulkdownloads/file/x.parquet",
                       "size": 0}],
            "schema": {"version": 1},
        }}

    def download(url, token, destination, timeout):
        calls["files"] += 1
        _sep_frame().to_parquet(destination, index=False)
        return destination.stat().st_size

    client = NasdaqBulkClient("never-store-me", json_getter=get_json, file_downloader=download)
    manifest = fetch_snapshot(
        client,
        [part],
        snapshot="unit-test",
        package="SEP",
        license_expires="2099-01-01",
        license_evidence=license_evidence,
        raw_root=raw_root,
        staged_root=staged_root,
    )
    assert calls == {"jobs": 1, "files": 1}
    assert "never-store-me" not in manifest.read_text()

    fetch_snapshot(
        client,
        [part],
        snapshot="unit-test",
        package="SEP",
        license_expires="2099-01-01",
        license_evidence=license_evidence,
        raw_root=raw_root,
        staged_root=staged_root,
    )
    assert calls == {"jobs": 1, "files": 1}

    extension = DownloadPart("SEP", 1, (("ticker.in[]", "BBB"),))
    with pytest.raises(SharadarError, match="plan is immutable"):
        fetch_snapshot(
            client,
            [part, extension],
            snapshot="unit-test",
            package="SEP",
            license_expires="2099-01-01",
            license_evidence=license_evidence,
            raw_root=raw_root,
            staged_root=staged_root,
        )
    assert calls == {"jobs": 1, "files": 1}
    completed = json.loads(manifest.read_text())
    assert completed["status"] == "COMPLETE"
    assert len(completed["planned_parts"]) == 1


def test_interrupted_snapshot_plan_cannot_be_replaced(tmp_path):
    evidence = tmp_path / "license.txt"
    evidence.write_text("written confirmation")
    first = DownloadPart("SEP", 0, (("ticker.in[]", "AAA"),))
    second = DownloadPart("SEP", 1, (("ticker.in[]", "BBB"),))
    replacement = DownloadPart("SEP", 1, (("ticker.in[]", "CCC"),))

    def get_json(url, token, timeout):
        if "BBB" in url:
            raise SharadarAPIError("terminal fixture", status_code=400)
        return {"bulk_download": {
            "status": "SUCCEEDED",
            "files": [{
                "url": "https://data.nasdaq.com/api/v1/bulkdownloads/file/x.parquet",
                "size": 0,
            }],
        }}

    def download(url, token, destination, timeout):
        _sep_frame().to_parquet(destination, index=False)
        return destination.stat().st_size

    client = NasdaqBulkClient("secret", json_getter=get_json, file_downloader=download)
    kwargs = {
        "snapshot": "interrupted",
        "package": "SEP",
        "license_expires": "2099-01-01",
        "license_evidence": evidence,
        "raw_root": tmp_path / "raw",
        "staged_root": tmp_path / "staged",
    }
    with pytest.raises(SharadarAPIError, match="terminal fixture"):
        fetch_snapshot(client, [first, second], **kwargs)
    manifest = json.loads(
        (tmp_path / "raw" / "interrupted" / "manifest.json").read_text()
    )
    assert manifest["status"] == "IN_PROGRESS"
    assert len(manifest["planned_parts"]) == 2
    with pytest.raises(SharadarError, match="plan is immutable"):
        fetch_snapshot(client, [first, replacement], **kwargs)


def test_empty_export_completes_and_tampering_fails_closed(tmp_path):
    evidence = tmp_path / "license.txt"
    evidence.write_text("written confirmation")
    client = NasdaqBulkClient(
        "secret",
        json_getter=lambda url, token, timeout: {
            "bulk_download": {"status": "SUCCEEDED", "files": []}
        },
    )
    manifest_path = fetch_snapshot(
        client,
        [DownloadPart("ACTIONS", 0, ())],
        snapshot="empty",
        package="SEP",
        license_expires="2099-01-01",
        license_evidence=evidence,
        raw_root=tmp_path / "raw",
        staged_root=tmp_path / "staged",
    )
    manifest = verify_snapshot_manifest(
        manifest_path,
        raw_snapshot=tmp_path / "raw" / "empty",
        staged_snapshot=tmp_path / "staged" / "empty",
    )
    assert manifest["parts"][0]["empty_export"] is True

    complete = _write_complete_snapshot(
        tmp_path / "raw" / "tampered",
        tmp_path / "staged" / "tampered",
        {"SEP": _sep_frame()},
    )
    raw_file = next((tmp_path / "raw" / "tampered").rglob("*.parquet"))
    raw_file.write_bytes(raw_file.read_bytes() + b"tamper")
    with pytest.raises(SharadarError, match="checksum verification failed"):
        verify_snapshot_manifest(
            complete,
            raw_snapshot=tmp_path / "raw" / "tampered",
            staged_snapshot=tmp_path / "staged" / "tampered",
            enforce_license=False,
        )


def test_snapshot_id_and_package_entitlements_fail_closed(tmp_path):
    evidence = tmp_path / "license.txt"
    evidence.write_text("written confirmation")
    client = NasdaqBulkClient("secret")
    with pytest.raises(ValueError, match="single safe component"):
        fetch_snapshot(
            client,
            [DownloadPart("SEP", 0, ())],
            snapshot="/tmp/escape",
            package="SEP",
            license_expires="2099-01-01",
            license_evidence=evidence,
            raw_root=tmp_path / "raw",
            staged_root=tmp_path / "staged",
        )
    with pytest.raises(ValueError, match="does not entitle"):
        fetch_snapshot(
            client,
            [DownloadPart("SF1", 0, ())],
            snapshot="bad-entitlement",
            package="SEP",
            license_expires="2099-01-01",
            license_evidence=evidence,
            raw_root=tmp_path / "raw",
            staged_root=tmp_path / "staged",
        )


def test_snapshot_symlinks_and_concurrent_lock_fail_closed(tmp_path):
    evidence = tmp_path / "license.txt"
    evidence.write_text("written confirmation")
    raw_root = tmp_path / "raw"
    staged_root = tmp_path / "staged"
    raw_root.mkdir()
    (raw_root / "real").mkdir()
    (raw_root / "alias").symlink_to(raw_root / "real", target_is_directory=True)
    client = NasdaqBulkClient("secret")
    kwargs = {
        "package": "SEP",
        "license_expires": "2099-01-01",
        "license_evidence": evidence,
        "raw_root": raw_root,
        "staged_root": staged_root,
    }
    with pytest.raises(SharadarError, match="symlink"):
        fetch_snapshot(
            client, [DownloadPart("SEP", 0, ())], snapshot="alias", **kwargs
        )

    external = tmp_path / "external"
    external.mkdir()
    (raw_root / "table-link").mkdir()
    (raw_root / "table-link" / "SEP").symlink_to(
        external, target_is_directory=True
    )
    with pytest.raises(SharadarError, match="symlinked snapshot entry"):
        fetch_snapshot(
            client, [DownloadPart("SEP", 0, ())], snapshot="table-link", **kwargs
        )

    with (raw_root / ".locked.lock").open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SharadarError, match="locked by another process"):
            fetch_snapshot(
                client, [DownloadPart("SEP", 0, ())], snapshot="locked", **kwargs
            )
    empty_client = NasdaqBulkClient(
        "secret",
        json_getter=lambda url, token, timeout: {
            "bulk_download": {"status": "SUCCEEDED", "files": []}
        },
    )
    manifest = fetch_snapshot(
        empty_client,
        [DownloadPart("ACTIONS", 0, ())],
        snapshot="locked",
        **kwargs,
    )
    assert json.loads(manifest.read_text())["status"] == "COMPLETE"


def test_purge_uses_manifest_tracked_roots(tmp_path):
    raw_snapshot = tmp_path / "raw" / "purge-me"
    staged_snapshot = tmp_path / "staged" / "purge-me"
    _write_complete_snapshot(
        raw_snapshot,
        staged_snapshot,
        {"SEP": _sep_frame()},
    )
    qa_snapshot = tmp_path / "qa" / "purge-me"
    qa_snapshot.mkdir(parents=True)
    (qa_snapshot / "run.json").write_text("{}")
    certificate = purge_snapshot_data(
        "purge-me",
        raw_root=tmp_path / "raw",
        staged_root=tmp_path / "staged",
        qa_root=tmp_path / "qa",
        certificate_root=tmp_path / "certificates",
    )
    assert certificate.exists()
    assert not raw_snapshot.exists()
    assert not staged_snapshot.exists()
    assert not qa_snapshot.exists()


def test_expired_license_fails_closed():
    with pytest.raises(SharadarError, match="license expired"):
        validate_license_expiry("2020-01-01", today=date(2020, 1, 2))


def test_raw_membership_preserves_parallel_share_classes(tmp_path):
    path = tmp_path / "membership.csv"
    pd.DataFrame({
        "date": ["2020-01-01"],
        "tickers": ["DISCA,DISCK,GOOG,GOOGL"],
    }).to_csv(path, index=False)
    snapshots = load_reference_snapshots_raw(path, xs_band=(4, 4))
    assert snapshots[0][1] == frozenset({"DISCA", "DISCK", "GOOG", "GOOGL"})


def test_identity_resolution_uses_dated_ticker_reuse():
    snapshots = [
        (pd.Timestamp("2019-01-01"), frozenset({"IR"})),
        (pd.Timestamp("2020-03-01"), frozenset()),
        (pd.Timestamp("2021-01-01"), frozenset({"IR"})),
    ]
    intervals = membership_intervals(
        snapshots,
        start=pd.Timestamp("2019-01-01"),
        end=pd.Timestamp("2021-12-31"),
    )
    resolved = resolve_identity_intervals(intervals, prepare_vendor_tickers(_ticker_frame()))
    assert list(resolved["match_status"]) == ["resolved", "resolved"]
    assert list(resolved["vendor_id"]) == ["101", "202"]


def test_identity_resolution_requires_complete_dated_coverage():
    intervals = pd.DataFrame({
        "interval_id": ["OLD:1"],
        "reference_ticker": ["OLD"],
        "valid_from": pd.to_datetime(["2010-01-01"]),
        "valid_to": pd.to_datetime(["2020-12-31"]),
    })
    partial = prepare_vendor_tickers(pd.DataFrame({
        "table": ["SEP"],
        "permaticker": [1],
        "ticker": ["OLD"],
        "name": ["Old"],
        "firstpricedate": pd.to_datetime(["2019-01-01"]),
        "lastpricedate": pd.to_datetime(["2021-01-01"]),
    }))
    assert resolve_identity_intervals(intervals, partial).iloc[0]["match_status"] == "unmatched"

    partial.loc[0, "firstpricedate"] = pd.NaT
    partial.loc[0, "bounds_complete"] = False
    assert resolve_identity_intervals(intervals, partial).iloc[0]["match_status"] == "unmatched"

    successor = prepare_vendor_tickers(pd.DataFrame({
        "table": ["SEP"],
        "permaticker": [2],
        "ticker": ["NEW"],
        "name": ["Successor"],
        "firstpricedate": pd.to_datetime(["2009-01-01"]),
        "lastpricedate": pd.to_datetime(["2021-01-01"]),
    }))
    override = pd.DataFrame({
        "reference_ticker": ["OLD"],
        "reference_from": pd.to_datetime(["2015-01-01"]),
        "reference_to": pd.to_datetime(["2020-12-31"]),
        "vendor_ticker": ["NEW"],
        "vendor_id": ["2"],
        "status": ["approved"],
        "evidence": ["dated filing"],
    })
    assert resolve_identity_intervals(
        intervals, successor, override
    ).iloc[0]["match_status"] == "unmatched"


def test_identity_overrides_are_bound_to_snapshot_and_inputs(tmp_path):
    path = tmp_path / "overrides.csv"
    pd.DataFrame({
        "reference_ticker": ["OLD"],
        "reference_from": ["2010-01-01"],
        "reference_to": ["2020-01-01"],
        "vendor_ticker": ["NEW"],
        "vendor_id": ["2"],
        "status": ["approved"],
        "evidence": ["filing"],
        "snapshot": ["snap-1"],
        "tickers_sha256": ["tickers"],
        "membership_sha256": ["membership"],
    }).to_csv(path, index=False)
    stale = load_identity_overrides(
        path,
        snapshot="snap-2",
        tickers_sha256="tickers",
        membership_sha256="membership",
    )
    assert not bool(stale.iloc[0]["binding_valid"])
    current = load_identity_overrides(
        path,
        snapshot="snap-1",
        tickers_sha256="tickers",
        membership_sha256="membership",
    )
    assert bool(current.iloc[0]["binding_valid"])


def test_panel_and_reference_aliases_compare_by_vendor_identity():
    metadata = prepare_vendor_tickers(pd.DataFrame({
        "table": ["SEP", "SEP"],
        "permaticker": [1, 1],
        "ticker": ["FB", "META"],
        "name": ["Meta", "Meta"],
        "firstpricedate": pd.to_datetime(["2010-01-01", "2022-06-09"]),
        "lastpricedate": pd.to_datetime(["2022-06-08", "2030-01-01"]),
    }))
    panel_ids = resolve_panel_identities(
        pd.DataFrame({"ticker": ["META"], "date": pd.to_datetime(["2024-01-02"])}),
        metadata,
    )
    assert panel_ids.iloc[0]["vendor_id"] == "1"
    crosswalk = pd.DataFrame({
        "interval_id": ["FB:1"],
        "reference_ticker": ["FB"],
        "valid_from": pd.to_datetime(["2010-01-01"]),
        "valid_to": pd.to_datetime(["2022-06-08"]),
        "vendor_ticker": ["FB"],
        "vendor_id": ["1"],
        "match_status": ["resolved"],
    })
    coverage = pd.DataFrame({
        "interval_id": ["FB:1"],
        "expected_member_days": [100],
        "observed_price_days": [100],
        "coverage": [1.0],
    })
    departed, summary = departed_target_coverage(crosswalk, coverage, {"1"})
    assert departed.empty
    assert summary["target"] == 0
    assert summary["coverage"] == 1.0


def test_price_and_departed_coverage_use_dynamic_target():
    crosswalk = pd.DataFrame({
        "interval_id": ["AAA:1", "OLD:1"],
        "reference_ticker": ["AAA", "OLD"],
        "valid_from": pd.to_datetime(["2020-01-02", "2020-01-02"]),
        "valid_to": pd.to_datetime(["2020-01-03", "2020-01-03"]),
        "vendor_ticker": ["AAA", "OLD"],
        "vendor_id": ["1", "2"],
        "match_status": ["resolved", "resolved"],
    })
    prices = pd.DataFrame({
        "ticker": ["AAA", "AAA", "OLD"],
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02"]),
        "open": [1.0, 1.0, 1.0],
        "high": [1.0, 1.0, 1.0],
        "low": [1.0, 1.0, 1.0],
        "close": [1.0, 1.0, 1.0],
        "closeadj": [1.0, 1.0, 1.0],
        "closeunadj": [1.0, 1.0, 1.0],
        "volume": [1.0, 1.0, 1.0],
        "dividends": [0.0, 0.0, 0.0],
    })
    calendar = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03"]))
    intervals, by_year, summary = price_coverage(crosswalk, prices, calendar)
    departed, target = departed_target_coverage(crosswalk, intervals, {"1"})
    assert summary["coverage"] == pytest.approx(0.75)
    assert by_year.iloc[0]["coverage"] == pytest.approx(0.75)
    assert target == {
        "denominator_kind": (
            "unique listing identities absent from the current panel; "
            "unresolved reference intervals remain failed targets"
        ),
        "interval_coverage_min": 0.98,
        "target": 1,
        "covered": 0,
        "coverage": 0.0,
    }
    assert departed.iloc[0]["reference_tickers"] == "OLD"

    unresolved = crosswalk.iloc[[1]].copy()
    unresolved["match_status"] = "unmatched"
    unresolved["vendor_id"] = None
    unresolved_status, unresolved_summary = departed_target_coverage(
        unresolved,
        intervals[intervals["interval_id"] == "OLD:1"],
        set(),
    )
    assert len(unresolved_status) == 1
    assert unresolved_summary["covered"] == 0

    invalid_prices = prices.copy()
    invalid_prices.loc[invalid_prices["ticker"] == "OLD", "closeunadj"] = None
    _, _, invalid_summary = price_coverage(crosswalk, invalid_prices, calendar)
    assert invalid_summary["coverage"] == pytest.approx(0.5)


def test_membership_reconciliation_rolls_events_backward():
    sp500 = pd.DataFrame({
        "date": pd.to_datetime(["2020-12-31", "2020-06-01", "2020-06-01"]),
        "ticker": ["NEW", "NEW", "OLD"],
        "action": ["current", "added", "removed"],
    })
    reference = [
        (pd.Timestamp("2020-01-31"), frozenset({"OLD"})),
        (pd.Timestamp("2020-06-30"), frozenset({"NEW"})),
    ]
    metadata = prepare_vendor_tickers(pd.DataFrame({
        "table": ["SEP", "SEP"],
        "permaticker": [1, 2],
        "ticker": ["OLD", "NEW"],
        "name": ["Old", "New"],
        "firstpricedate": pd.to_datetime(["2010-01-01", "2020-06-01"]),
        "lastpricedate": pd.to_datetime(["2020-05-31", "2025-01-01"]),
    }))
    out = membership_reconciliation(
        sp500,
        reference,
        metadata,
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-06-30"),
    )
    assert list(out["date"]) == list(pd.date_range("2020-01-31", "2020-06-30", freq="ME"))
    assert list(out["jaccard"]) == [1.0] * 6


def test_membership_jaccard_penalizes_unresolved_symbols():
    sp500 = pd.DataFrame({
        "date": pd.to_datetime(["2020-12-31", "2020-12-31"]),
        "ticker": ["AAA", "VENDOR_MISSING"],
        "action": ["current", "current"],
    })
    reference = [(
        pd.Timestamp("2020-01-31"),
        frozenset({"AAA", "REFERENCE_MISSING"}),
    )]
    metadata = prepare_vendor_tickers(pd.DataFrame({
        "table": ["SEP"],
        "permaticker": [1],
        "ticker": ["AAA"],
        "name": ["Alpha"],
        "firstpricedate": pd.to_datetime(["2010-01-01"]),
        "lastpricedate": pd.to_datetime(["2030-01-01"]),
    }))
    out = membership_reconciliation(
        sp500,
        reference,
        metadata,
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-31"),
    )
    assert out.iloc[0]["unresolved_reference"] == 1
    assert out.iloc[0]["unresolved_vendor"] == 1
    assert out.iloc[0]["jaccard"] == pytest.approx(1 / 3)


def test_persistent_membership_difference_requires_adjudication():
    dates = pd.bdate_range("2020-01-02", periods=6)
    daily = pd.DataFrame({
        "date": dates,
        "reference_only_ids": ["101"] * 6,
        "vendor_only_ids": [""] * 6,
    })
    report = membership_difference_report(
        daily,
        None,
        snapshot="snap",
        tickers_sha256="tickers",
        membership_sha256="membership",
    )
    assert len(report) == 1
    assert report.iloc[0]["trading_days"] == 6
    assert report.iloc[0]["manual_status"] == "review_required"


def test_calendar_quality_catches_concentrated_holes():
    full = pd.bdate_range("2020-01-01", "2020-12-31")
    missing_march = full[full.month != 3]
    quality = calendar_quality(
        missing_march,
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-12-31"),
    )
    assert quality["min_month_weekday_coverage"] == 0.0
    assert quality["max_internal_gap_days"] > 7


def test_known_cases_require_evidenced_manual_approval(tmp_path):
    metadata = prepare_vendor_tickers(_ticker_frame())
    cases = {"cases": [{
        "case_id": "ir",
        "symbols": ["IR"],
        "expectation": "ticker reuse is separated",
    }]}
    pending = known_case_report(
        metadata,
        cases,
        None,
        snapshot="snap-1",
        tickers_sha256="ticker-hash",
        cases_sha256="cases-hash",
    )
    assert pending.iloc[0]["manual_status"] == "review_required"

    approvals = tmp_path / "approvals.csv"
    pd.DataFrame({
        "case_id": ["ir"],
        "status": ["approved"],
        "evidence": ["vendor rows plus corporate-action filing"],
        "snapshot": ["snap-1"],
        "observed_ids_sha256": [pending.iloc[0]["observed_ids_sha256"]],
        "tickers_sha256": ["ticker-hash"],
        "cases_sha256": ["cases-hash"],
    }).to_csv(approvals, index=False)
    approved = known_case_report(
        metadata,
        cases,
        approvals,
        snapshot="snap-1",
        tickers_sha256="ticker-hash",
        cases_sha256="cases-hash",
    )
    assert approved.iloc[0]["manual_status"] == "approved"

    stale = known_case_report(
        metadata,
        cases,
        approvals,
        snapshot="snap-2",
        tickers_sha256="ticker-hash",
        cases_sha256="cases-hash",
    )
    assert stale.iloc[0]["approval_reason"] == "snapshot_mismatch"

    frame = pd.read_csv(approvals)
    frame.loc[0, "evidence"] = None
    frame.to_csv(approvals, index=False)
    blank = known_case_report(
        metadata,
        cases,
        approvals,
        snapshot="snap-1",
        tickers_sha256="ticker-hash",
        cases_sha256="cases-hash",
    )
    assert blank.iloc[0]["approval_reason"] == "blank_evidence"


def test_blind_qa_end_to_end_synthetic_snapshot(tmp_path):
    names = [f"T{i:03d}" for i in range(490)]
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    snapshot = "synthetic"
    raw_root = tmp_path / "raw"
    staged_root = tmp_path / "staged"
    raw_snapshot = raw_root / snapshot
    staged_snapshot = staged_root / snapshot

    membership = tmp_path / "membership.csv"
    pd.DataFrame({
        "date": dates,
        "tickers": [",".join(names), ",".join(names)],
    }).to_csv(membership, index=False)
    panel = tmp_path / "panel.parquet"
    panel_frame = pd.DataFrame([
        {"ticker": ticker, "date": d}
        for ticker in names[:-1]
        for d in dates[:1]
    ])
    panel_frame = pd.concat([
        panel_frame,
        pd.DataFrame({"ticker": [names[0]], "date": [dates[1]]}),
    ], ignore_index=True)
    panel_frame.to_parquet(panel, index=False)

    tickers = pd.DataFrame({
        "table": "SEP",
        "permaticker": range(1, 491),
        "ticker": names,
        "name": names,
        "firstpricedate": pd.Timestamp("2010-01-01"),
        "lastpricedate": pd.Timestamp("2030-01-01"),
    })
    prices = pd.DataFrame([
        {
            "ticker": ticker,
            "date": d,
            "open": 10.0,
            "high": 10.2,
            "low": 9.9,
            "close": 10.1,
            "volume": 1000,
            "closeadj": 10.1,
            "closeunadj": 10.1,
            "dividends": 0.0,
            "lastupdated": dates[-1],
        }
        for ticker in names
        for d in dates
    ])
    sp500 = pd.DataFrame({
        "date": dates[-1],
        "ticker": names,
        "action": "current",
    })
    actions = pd.DataFrame({
        "date": [dates[0]],
        "ticker": [names[0]],
        "name": [names[0]],
        "action": ["dividend"],
        "contraname": [None],
        "contraticker": [None],
    })
    _write_complete_snapshot(raw_snapshot, staged_snapshot, {
        "TICKERS": tickers,
        "SEP": prices,
        "SP500": sp500,
        "ACTIONS": actions,
    })

    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({
        "schema_version": 1,
        "departed_identity_coverage_min": 1.0,
        "departed_interval_price_coverage_min": 1.0,
        "member_day_price_coverage_min": 1.0,
        "member_day_price_coverage_by_year_min": 1.0,
        "membership_month_end_jaccard_min": 1.0,
        "cross_section_min": 490,
        "cross_section_max": 515,
        "unresolved_primary_key_duplicates_max": 0,
        "unresolved_identity_ambiguities_max": 0,
        "known_case_approval_rate_min": 0.0,
        "identity_registry_coverage_min": 0.0,
        "calendar_weekday_coverage_min": 1.0,
        "calendar_month_weekday_coverage_min": 1.0,
        "membership_difference_approval_rate_min": 1.0,
        "calendar_endpoint_gap_days_max": 0,
        "calendar_internal_gap_days_max": 1,
        "calendar_expected_dates": 2,
        "calendar_expected_sha256": hashlib.sha256(
            b"2020-01-02\n2020-01-03\n"
        ).hexdigest(),
    }))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"schema_version": 1, "cases": []}))
    output = tmp_path / "qa"
    run = audit_snapshot(
        snapshot=snapshot,
        staged_root=staged_root,
        raw_root=raw_root,
        membership_csv=membership,
        panel_path=panel,
        thresholds_path=thresholds,
        cases_path=cases,
        output_dir=output,
        start="2020-01-02",
        enforce_license=False,
    )
    assert run["status"] == "NONCANONICAL"
    assert run["gate_status"] == "PASS"
    assert run["departed_target"]["target"] == 1
    assert run["departed_target"]["covered"] == 1
    assert run["price_coverage"]["calendar_dates"] == 2
    assert (output / "run.json").exists()
    assert "Status: **NONCANONICAL**" in (output / "report.md").read_text()

    with pytest.raises(SharadarError, match="must not be inside raw or staged"):
        audit_snapshot(
            snapshot=snapshot,
            staged_root=staged_root,
            raw_root=raw_root,
            membership_csv=membership,
            panel_path=panel,
            thresholds_path=thresholds,
            cases_path=cases,
            output_dir=raw_snapshot,
            start="2020-01-02",
            enforce_license=False,
        )
