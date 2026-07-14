"""Synthetic tests for staged Sharadar ingestion and blind identity QA."""

from __future__ import annotations

import json
import hashlib
import fcntl
import shutil
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
    price_jump_review,
    resolve_panel_identities,
    resolve_identity_intervals,
    resolve_sep_rows,
    staged_table_sha256,
    unexplained_price_jumps,
    verify_price_jumps,
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


def test_purge_resumes_from_journal_after_interruption(tmp_path, monkeypatch):
    import alpha_graph.data.sharadar as sharadar

    raw_snapshot = tmp_path / "raw" / "purge-me"
    staged_snapshot = tmp_path / "staged" / "purge-me"
    _write_complete_snapshot(raw_snapshot, staged_snapshot, {"SEP": _sep_frame()})
    qa_snapshot = tmp_path / "qa" / "purge-me"
    qa_snapshot.mkdir(parents=True)
    (qa_snapshot / "run.json").write_text("{}")
    kwargs = {
        "raw_root": tmp_path / "raw",
        "staged_root": tmp_path / "staged",
        "qa_root": tmp_path / "qa",
        "certificate_root": tmp_path / "certificates",
    }

    real_rmtree = shutil.rmtree
    calls = {"count": 0}

    def dying_rmtree(path, *args, **kw):
        if calls["count"] >= 1:
            raise OSError("simulated crash")
        calls["count"] += 1
        return real_rmtree(path, *args, **kw)

    monkeypatch.setattr(sharadar.shutil, "rmtree", dying_rmtree)
    with pytest.raises(OSError, match="simulated crash"):
        purge_snapshot_data("purge-me", **kwargs)
    journal_path = tmp_path / "certificates" / "purge-me.journal.json"
    journal = json.loads(journal_path.read_text())
    assert journal["status"] == "IN_PROGRESS"
    assert journal["phases"] == {"qa": "DELETED", "staged": "PENDING", "raw": "PENDING"}
    assert not qa_snapshot.exists()
    assert staged_snapshot.exists()
    assert raw_snapshot.exists()
    assert not (tmp_path / "certificates" / "purge-me.json").exists()

    monkeypatch.setattr(sharadar.shutil, "rmtree", real_rmtree)
    certificate = purge_snapshot_data("purge-me", **kwargs)
    assert certificate.exists()
    assert not raw_snapshot.exists()
    assert not staged_snapshot.exists()
    payload = json.loads(certificate.read_text())
    assert payload["deleted_file_count"] == 5
    assert payload["deleted_file_count"] == len(payload["deleted_inventory"])
    journal = json.loads(journal_path.read_text())
    assert journal["status"] == "COMPLETE"
    assert journal["phases"] == {"qa": "DELETED", "staged": "DELETED", "raw": "DELETED"}


def test_purge_completes_from_journal_when_raw_is_gone(tmp_path):
    certificate_root = tmp_path / "certificates"
    certificate_root.mkdir()
    journal = {
        "schema_version": 1,
        "snapshot": "orphan",
        "created_at_utc": "2020-01-01T00:00:00+00:00",
        "storage_roots": {
            "raw": str((tmp_path / "raw").resolve()),
            "staged": str((tmp_path / "staged").resolve()),
            "qa": str((tmp_path / "qa").resolve()),
        },
        "manifest_sha256_before_purge": "fixture-manifest-sha",
        "deleted_inventory": [
            {"kind": "raw", "relative_path": "manifest.json", "bytes": 10},
            {"kind": "staged", "relative_path": "SEP/data.parquet", "bytes": 20},
        ],
        "phases": {"qa": "DELETED", "staged": "DELETED", "raw": "PENDING"},
        "status": "IN_PROGRESS",
    }
    (certificate_root / "orphan.journal.json").write_text(json.dumps(journal))
    # The raw root (and its manifest) is already gone; the missing-manifest
    # guard must not block journal-driven completion.
    certificate = purge_snapshot_data(
        "orphan",
        raw_root=tmp_path / "raw",
        staged_root=tmp_path / "staged",
        qa_root=tmp_path / "qa",
        certificate_root=certificate_root,
    )
    payload = json.loads(certificate.read_text())
    assert payload["manifest_sha256_before_purge"] == "fixture-manifest-sha"
    assert payload["deleted_file_count"] == 2
    assert payload["deleted_bytes"] == 30
    final = json.loads((certificate_root / "orphan.journal.json").read_text())
    assert final["status"] == "COMPLETE"
    assert final["phases"]["raw"] == "DELETED"


def test_license_json_recreated_on_in_progress_resume(tmp_path):
    evidence = tmp_path / "license.txt"
    evidence.write_text("written confirmation")
    first = DownloadPart("SEP", 0, (("ticker.in[]", "AAA"),))
    second = DownloadPart("SEP", 1, (("ticker.in[]", "BBB"),))
    fail_second = {"flag": True}

    def get_json(url, token, timeout):
        if "BBB" in url and fail_second["flag"]:
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
        "snapshot": "resume-license",
        "package": "SEP",
        "license_expires": "2099-01-01",
        "license_evidence": evidence,
        "raw_root": tmp_path / "raw",
        "staged_root": tmp_path / "staged",
    }
    with pytest.raises(SharadarAPIError, match="terminal fixture"):
        fetch_snapshot(client, [first, second], **kwargs)
    license_path = tmp_path / "raw" / "resume-license" / "license.json"
    assert license_path.exists()
    license_path.unlink()

    fail_second["flag"] = False
    manifest_path = fetch_snapshot(client, [first, second], **kwargs)
    assert json.loads(manifest_path.read_text())["status"] == "COMPLETE"
    recreated = json.loads(license_path.read_text())
    assert recreated["snapshot"] == "resume-license"
    assert recreated["package"] == "SEP"
    assert recreated["written_confirmation_sha256"] == sha256_file(evidence)

    recreated["license_expires"] = "2098-01-01"
    license_path.write_text(json.dumps(recreated))
    with pytest.raises(SharadarError, match="license metadata mismatch"):
        fetch_snapshot(client, [first, second], **kwargs)


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


def test_raw_membership_band_applies_only_from_band_from(tmp_path):
    # The real CSV has pre-window (1996) rows below the modern band; they must
    # still LOAD (they accumulate the state at the audit start) while the band
    # check applies only inside the audited era.
    path = tmp_path / "membership.csv"
    pd.DataFrame({
        "date": ["1996-01-02", "2020-01-01"],
        "tickers": ["A,B", "A,B,C,D"],
    }).to_csv(path, index=False)
    snapshots = load_reference_snapshots_raw(
        path, xs_band=(4, 4), band_from=pd.Timestamp("2011-04-06"),
    )
    assert len(snapshots) == 2                       # pre-window row loaded
    assert snapshots[0][1] == frozenset({"A", "B"})  # and usable for state
    # an IN-window violation still fails closed
    bad = tmp_path / "bad.csv"
    pd.DataFrame({
        "date": ["1996-01-02", "2020-01-01"],
        "tickers": ["A,B", "A,B,C"],
    }).to_csv(bad, index=False)
    with pytest.raises(SharadarError, match="outside"):
        load_reference_snapshots_raw(
            bad, xs_band=(4, 4), band_from=pd.Timestamp("2011-04-06"),
        )
    # and without band_from the old strict-everywhere behavior is unchanged
    with pytest.raises(SharadarError, match="outside"):
        load_reference_snapshots_raw(path, xs_band=(4, 4))


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


def _synthetic_audit_kwargs(
    tmp_path,
    *,
    names_count=3,
    threshold_overrides=None,
    tickers=None,
    prices=None,
    actions=None,
):
    """Build a complete synthetic snapshot plus audit inputs under tmp_path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    names = [f"T{i:03d}" for i in range(names_count)]
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    snapshot = "synthetic"
    raw_root = tmp_path / "raw"
    staged_root = tmp_path / "staged"

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

    if tickers is None:
        tickers = pd.DataFrame({
            "table": "SEP",
            "permaticker": range(1, names_count + 1),
            "ticker": names,
            "name": names,
            "firstpricedate": pd.Timestamp("2010-01-01"),
            "lastpricedate": pd.Timestamp("2030-01-01"),
        })
    if prices is None:
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
    if actions is None:
        actions = pd.DataFrame({
            "date": [dates[0]],
            "ticker": [names[0]],
            "name": [names[0]],
            "action": ["dividend"],
            "contraname": [None],
            "contraticker": [None],
        })
    _write_complete_snapshot(raw_root / snapshot, staged_root / snapshot, {
        "TICKERS": tickers,
        "SEP": prices,
        "SP500": sp500,
        "ACTIONS": actions,
    })

    thresholds_value = {
        "schema_version": 1,
        "departed_identity_coverage_min": 1.0,
        "departed_interval_price_coverage_min": 1.0,
        "member_day_price_coverage_min": 1.0,
        "member_day_price_coverage_by_year_min": 1.0,
        "membership_month_end_jaccard_min": 1.0,
        "cross_section_min": names_count,
        "cross_section_max": max(names_count, 515),
        "unresolved_primary_key_duplicates_max": 0,
        "unresolved_identity_ambiguities_max": 0,
        "sep_quarantined_rows_max": 0,
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
    }
    thresholds_value.update(threshold_overrides or {})
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps(thresholds_value))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"schema_version": 1, "cases": []}))
    return {
        "snapshot": snapshot,
        "staged_root": staged_root,
        "raw_root": raw_root,
        "membership_csv": membership,
        "panel_path": panel,
        "thresholds_path": thresholds,
        "cases_path": cases,
        "output_dir": tmp_path / "qa",
        "start": "2020-01-02",
        "enforce_license": False,
    }


def test_blind_qa_end_to_end_synthetic_snapshot(tmp_path):
    kwargs = _synthetic_audit_kwargs(tmp_path, names_count=490)
    output = kwargs["output_dir"]
    run = audit_snapshot(**kwargs)
    assert run["status"] == "NONCANONICAL"
    assert run["gate_status"] == "PASS"
    assert run["departed_target"]["target"] == 1
    assert run["departed_target"]["covered"] == 1
    assert run["price_coverage"]["calendar_dates"] == 2
    assert (output / "run.json").exists()
    assert "Status: **NONCANONICAL**" in (output / "report.md").read_text()

    with pytest.raises(SharadarError, match="must not be inside raw or staged"):
        audit_snapshot(**{
            **kwargs,
            "output_dir": kwargs["raw_root"] / kwargs["snapshot"],
        })


def test_resolve_sep_rows_quarantines_zero_and_multiple_listings():
    metadata = prepare_vendor_tickers(pd.DataFrame({
        "table": ["SEP", "SEP", "SEP"],
        "permaticker": [1, 2, 3],
        "ticker": ["AAA", "AAA", "BBB"],
        "name": ["Alpha one", "Alpha two", "Beta"],
        "firstpricedate": pd.to_datetime(["2010-01-01"] * 3),
        "lastpricedate": pd.to_datetime(["2030-01-01"] * 3),
    }))
    prices = pd.DataFrame({
        "ticker": ["AAA", "BBB", "GHOST", "BBB"],
        "date": pd.to_datetime(
            ["2020-01-02", "2020-01-02", "2020-01-02", "2030-01-20"]
        ),
    })
    resolved, quarantined = resolve_sep_rows(prices, metadata)
    assert list(resolved["ticker"]) == ["BBB"]
    assert resolved.iloc[0]["listing_key"] == "unverified:3"
    assert sorted(quarantined["quarantine_reason"]) == [
        "multiple_listings", "zero_listings", "zero_listings",
    ]
    ambiguous = quarantined[quarantined["ticker"] == "AAA"].iloc[0]
    assert ambiguous["candidate_listings"] == "unverified:1|unverified:2"

    registry_resolved, registry_quarantined = resolve_sep_rows(
        prices, metadata, identity_map={"1": "L-AAA", "2": "L-AAA"}
    )
    assert sorted(registry_resolved["listing_key"]) == ["L-AAA", "unverified:3"]
    assert list(registry_quarantined["ticker"]) == ["GHOST", "BBB"]


def test_audit_quarantines_ambiguous_sep_rows(tmp_path):
    tickers = pd.DataFrame({
        "table": "SEP",
        "permaticker": [1, 2, 3, 99],
        "ticker": ["T000", "T001", "T002", "T000"],
        "name": ["T000", "T001", "T002", "T000 duplicate"],
        "firstpricedate": pd.to_datetime(["2010-01-01"] * 4),
        "lastpricedate": pd.to_datetime(["2030-01-01"] * 4),
    })
    kwargs = _synthetic_audit_kwargs(tmp_path, tickers=tickers)
    run = audit_snapshot(**kwargs)
    assert run["gate_status"] == "FAIL"
    assert run["sep_row_resolution"]["quarantined_rows"] == 2
    assert run["sep_row_resolution"]["quarantined_multiple_listing_rows"] == 2
    quarantine = pd.read_parquet(kwargs["output_dir"] / "sep_quarantine.parquet")
    assert len(quarantine) == 2
    assert set(quarantine["ticker"]) == {"T000"}
    # Quarantined rows are excluded from the coverage numerator.
    assert run["price_coverage"]["expected_member_days"] == 6
    assert run["price_coverage"]["observed_price_days"] == 4
    issues = pd.read_parquet(kwargs["output_dir"] / "issues.parquet")
    quarantine_issue = issues[issues["gate"] == "sep_row_quarantine"]
    assert len(quarantine_issue) == 1
    assert quarantine_issue.iloc[0]["severity"] == "high"


def test_price_jump_math_verification_rejects_ordinary_dividends(tmp_path):
    prices = pd.DataFrame({
        "ticker": ["SPL", "SPL", "RSP", "RSP", "DIV", "DIV", "SPC", "SPC"],
        "date": pd.to_datetime(["2020-01-02", "2020-01-03"] * 4),
        "closeunadj": [100.0, 48.0, 10.0, 100.0, 100.0, 40.0, 100.0, 45.0],
        "dividends": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 55.0],
    })
    actions = pd.DataFrame({
        "ticker": ["SPL", "RSP", "DIV"],
        "date": pd.to_datetime(["2020-01-03"] * 3),
        "action": ["split", "split", "dividend"],
        "value": [2.0, 0.1, 0.5],
    })
    jumps = unexplained_price_jumps(prices, actions)
    assert sorted(jumps["ticker"]) == ["DIV", "RSP", "SPC", "SPL"]
    verified = verify_price_jumps(jumps, actions, prices).set_index("ticker")
    assert verified.loc["SPL", "verified_method"] == "split_math"
    assert verified.loc["RSP", "verified_method"] == "split_math"
    assert verified.loc["SPC", "verified_method"] == "distribution_math"
    # A nearby ordinary dividend must not clear a 60% move.
    assert not bool(verified.loc["DIV", "math_verified"])

    # Without the ACTIONS value column no split can be math-verified.
    valueless = verify_price_jumps(
        jumps, actions.drop(columns="value"), prices
    ).set_index("ticker")
    assert not bool(valueless.loc["SPL", "math_verified"])
    assert bool(valueless.loc["SPC", "math_verified"])

    reviewed = price_jump_review(
        verified.reset_index(),
        None,
        snapshot="snap-1",
        tickers_sha256="tick-hash",
    ).set_index("ticker")
    assert reviewed.loc["SPL", "manual_status"] == "auto_verified"
    assert reviewed.loc["DIV", "manual_status"] == "review_required"

    approvals = tmp_path / "jump_approvals.csv"
    pd.DataFrame({
        "jump_id": [reviewed.loc["DIV", "jump_id"]],
        "status": ["approved"],
        "evidence": ["verified against the exchange notice"],
        "snapshot": ["snap-1"],
        "observed_sha256": [reviewed.loc["DIV", "observed_sha256"]],
        "tickers_sha256": ["tick-hash"],
    }).to_csv(approvals, index=False)
    approved = price_jump_review(
        verified.reset_index(),
        approvals,
        snapshot="snap-1",
        tickers_sha256="tick-hash",
    ).set_index("ticker")
    assert approved.loc["DIV", "manual_status"] == "approved"
    stale = price_jump_review(
        verified.reset_index(),
        approvals,
        snapshot="snap-2",
        tickers_sha256="tick-hash",
    ).set_index("ticker")
    assert stale.loc["DIV", "manual_status"] == "review_required"


def test_audit_gates_unexplained_jump_until_bound_approval(tmp_path):
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])

    def price_row(ticker, day, closeunadj, dividends=0.0):
        return {
            "ticker": ticker,
            "date": day,
            "open": closeunadj,
            "high": closeunadj,
            "low": closeunadj,
            "close": closeunadj,
            "volume": 1000,
            "closeadj": 10.0,
            "closeunadj": closeunadj,
            "dividends": dividends,
            "lastupdated": dates[-1],
        }

    prices = pd.DataFrame([
        price_row("T000", dates[0], 10.0),
        price_row("T000", dates[1], 10.0),
        # True 2:1 split with a matching ACTIONS value: auto-verified.
        price_row("T001", dates[0], 100.0),
        price_row("T001", dates[1], 48.0),
        # 60% drop beside an ordinary dividend: requires manual approval.
        price_row("T002", dates[0], 100.0),
        price_row("T002", dates[1], 40.0, dividends=0.5),
    ])
    actions = pd.DataFrame({
        "date": [dates[1], dates[1]],
        "ticker": ["T001", "T002"],
        "name": ["T001", "T002"],
        "action": ["split", "dividend"],
        "contraname": [None, None],
        "contraticker": [None, None],
        "value": [2.0, 0.5],
    })
    kwargs = _synthetic_audit_kwargs(tmp_path, prices=prices, actions=actions)
    run = audit_snapshot(**kwargs)
    assert run["gate_status"] == "FAIL"
    assert run["price_jumps"] == {
        "total_large_moves": 2,
        "math_verified": 1,
        "approved": 0,
        "review_required": 1,
    }
    issues = pd.read_parquet(kwargs["output_dir"] / "issues.parquet")
    assert (issues["gate"] == "price_jump_review").any()
    inventory = pd.read_parquet(
        kwargs["output_dir"] / "all_large_price_jumps.parquet"
    )
    assert sorted(inventory["ticker"]) == ["T001", "T002"]
    unexplained = inventory[inventory["manual_status"] == "review_required"]
    assert list(unexplained["ticker"]) == ["T002"]

    approvals = tmp_path / "jump_approvals.csv"
    pd.DataFrame({
        "jump_id": [unexplained.iloc[0]["jump_id"]],
        "status": ["approved"],
        "evidence": ["special distribution confirmed by filing"],
        "snapshot": ["synthetic"],
        "observed_sha256": [unexplained.iloc[0]["observed_sha256"]],
        "tickers_sha256": [
            staged_table_sha256(kwargs["staged_root"] / "synthetic", "TICKERS")
        ],
    }).to_csv(approvals, index=False)
    cleared = audit_snapshot(**kwargs, jump_approvals_path=approvals)
    assert cleared["gate_status"] == "PASS"
    assert cleared["price_jumps"]["approved"] == 1
    assert cleared["price_jumps"]["review_required"] == 0


def test_audit_aborts_without_report_when_inputs_change_mid_run(tmp_path, monkeypatch):
    import alpha_graph.data.sharadar_qa as sharadar_qa

    kwargs = _synthetic_audit_kwargs(tmp_path)
    original = sharadar_qa.alias_window_audit

    def mutate_cases_then_run(vendor_tickers):
        kwargs["cases_path"].write_text(
            json.dumps({"schema_version": 1, "cases": []}, indent=2)
        )
        return original(vendor_tickers)

    monkeypatch.setattr(sharadar_qa, "alias_window_audit", mutate_cases_then_run)
    with pytest.raises(SharadarError, match="changed during the audit"):
        audit_snapshot(**kwargs)
    assert not (kwargs["output_dir"] / "run.json").exists()
    assert not (kwargs["output_dir"] / "report.md").exists()


def test_policy_hash_is_stable_and_recorded(tmp_path):
    first = _synthetic_audit_kwargs(tmp_path / "one")
    second = _synthetic_audit_kwargs(tmp_path / "two")
    run_one = audit_snapshot(**first)
    run_two = audit_snapshot(**second)
    assert run_one["gate_status"] == "PASS"
    assert len(run_one["policy_hash"]) == 64
    assert run_one["policy_hash"] == run_two["policy_hash"]
    assert isinstance(run_one["dirty"], bool)
    assert isinstance(run_one["dirty_path_count"], int)
    # Synthetic paths are never the canonical policy, so no canonical marker.
    assert run_one["status"] == "NONCANONICAL"
    assert run_one["canonical_pass"] is None
    report = (first["output_dir"] / "report.md").read_text()
    assert run_one["policy_hash"] in report
    saved = json.loads((first["output_dir"] / "run.json").read_text())
    assert saved["policy_hash"] == run_one["policy_hash"]
    assert saved["git_commit"] == run_one["git_commit"]


def test_audit_fails_on_calendar_expectation_mismatch(tmp_path):
    count_kwargs = _synthetic_audit_kwargs(
        tmp_path / "count",
        threshold_overrides={"calendar_expected_dates": 3},
    )
    run = audit_snapshot(**count_kwargs)
    assert run["gate_status"] == "FAIL"
    issues = pd.read_parquet(count_kwargs["output_dir"] / "issues.parquet")
    detail = issues[issues["gate"] == "calendar_expectation"]
    assert len(detail) == 1
    assert detail.iloc[0]["severity"] == "high"

    hash_kwargs = _synthetic_audit_kwargs(
        tmp_path / "hash",
        threshold_overrides={"calendar_expected_sha256": "0" * 64},
    )
    run = audit_snapshot(**hash_kwargs)
    assert run["gate_status"] == "FAIL"
    assert run["price_coverage"]["calendar_sha256"] == hashlib.sha256(
        b"2020-01-02\n2020-01-03\n"
    ).hexdigest()
    issues = pd.read_parquet(hash_kwargs["output_dir"] / "issues.parquet")
    assert (issues["gate"] == "calendar_expectation").any()
