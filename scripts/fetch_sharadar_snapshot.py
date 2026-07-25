"""Download whole Sharadar tables through the v3 datatable export API.

Written on 2026-07-25 to secure the raw vendor files while
``alpha_graph.data.sharadar`` was still calling a dead ``/api/v1/bulkdownloads``
endpoint. That module has since been repaired and is the supported path: it
produces the hash-verified snapshot and staged Parquet the QA gates consume,
which this script does not. Prefer ``python -m alpha_graph.data.sharadar fetch``.

This script remains only because the snapshot it already wrote is on disk in its
own layout, and because it can pull tables the module has no TableSpec for
(notably SFP fund prices). Its output is NOT a v2 snapshot: the audited module's
``purge`` and ``verify_snapshot_manifest`` cannot read it, so use ``--purge``
here to delete it when the licence lapses.

Verified transport (2026-07-25, live SFA key):

    GET {BASE}/{TABLE}.json?{filters}&qopts.export=true   X-Api-Token header
      -> {"datatable_bulk_download": {"file": {"link", "status",
                                               "data_snapshot_time"}}}
      status "fresh" means ready; "creating"/"regenerating" mean poll again.
    GET file.link (presigned S3, no auth header) -> ZIP containing one CSV.

Every table is written once, hashed, and recorded in ``manifest.json``. Re-runs
skip tables whose recorded hash still matches the file on disk, so an
interrupted download resumes without re-fetching what already landed.

Usage::

    python scripts/fetch_sharadar_snapshot.py --license-expires 2026-08-25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
API_HOST = "data.nasdaq.com"
DOWNLOAD_HOST_SUFFIX = ".s3.amazonaws.com"
DOWNLOAD_BUCKET_PREFIX = "aws-gis-link-"
KEY_FILE = Path(__file__).resolve().parent.parent / "data" / "private" / "nasdaq_api_key.txt"
RAW_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "sharadar"

# Whole-table pulls: the licence is one month, so take maximum history and
# maximum cross-section now and narrow locally afterwards.
TABLES = (
    "TICKERS",   # reference: listing identity, delisting flags, first/last dates
    "SP500",     # index membership events
    "ACTIONS",   # splits, dividends, delistings, ticker changes
    "SEP",       # daily equity prices, active and delisted
    "SFP",       # daily fund prices (ETF/CEF/ETN)
    "SF1",       # fundamentals
    "SF2",       # insider holdings (Form 4)
    "SF3",       # institutional holdings (13F)
    "EVENTS",    # 8-K event codes
    "DAILY",     # daily valuation metrics
)

POLL_SECONDS = 20.0
POLL_TIMEOUT_SECONDS = 3600.0
REQUEST_TIMEOUT = 180.0
DOWNLOAD_TIMEOUT = 3600.0


class FetchError(RuntimeError):
    pass


def _log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def _read_key() -> str:
    if not KEY_FILE.is_file():
        raise FetchError(f"missing API key file {KEY_FILE}")
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise FetchError("API key file is empty")
    return key


def _export_url(table: str) -> str:
    return f"{BASE}/{table}.json?{urllib.parse.urlencode({'qopts.export': 'true'})}"


def _get_json(url: str, token: str) -> dict:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != API_HOST:
        raise FetchError(f"refusing to call non-Nasdaq URL host {parsed.hostname!r}")
    request = urllib.request.Request(
        url, headers={"X-Api-Token": token, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:400]
        raise FetchError(f"HTTP {exc.code} from export API: {body}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise FetchError(f"transport error calling export API: {exc}") from exc


def _wait_for_link(table: str, token: str) -> dict:
    """Poll the export endpoint until the vendor reports a fresh file."""
    url = _export_url(table)
    started = time.monotonic()
    while True:
        payload = _get_json(url, token)
        job = payload.get("datatable_bulk_download")
        if not isinstance(job, dict) or not isinstance(job.get("file"), dict):
            raise FetchError(f"{table}: unexpected export response shape")
        file_info = job["file"]
        status = str(file_info.get("status", "")).lower()
        if status == "fresh":
            if not file_info.get("link"):
                raise FetchError(f"{table}: fresh export carries no link")
            return file_info
        if status in {"creating", "regenerating"}:
            if time.monotonic() - started >= POLL_TIMEOUT_SECONDS:
                raise FetchError(f"{table}: export stuck in status {status}")
            _log(f"{table}: status {status}, waiting {POLL_SECONDS:.0f}s")
            time.sleep(POLL_SECONDS)
            continue
        raise FetchError(f"{table}: unexpected export status {status!r}")


def _download(link: str, destination: Path) -> int:
    """Stream a presigned export to disk. The link carries its own auth."""
    parsed = urllib.parse.urlparse(link)
    host = parsed.hostname or ""
    # The presigned signature authenticates the vendor's object, not the host,
    # and any AWS customer can register a bucket.
    if (
        parsed.scheme != "https"
        or not host.endswith(DOWNLOAD_HOST_SUFFIX)
        or not host.startswith(DOWNLOAD_BUCKET_PREFIX)
    ):
        raise FetchError(f"refusing an export link outside the vendor bucket: {host!r}")
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(link, timeout=DOWNLOAD_TIMEOUT) as response, \
                partial.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        partial.unlink(missing_ok=True)
        raise FetchError(f"transport error downloading export: {exc}") from exc
    written = partial.stat().st_size
    if written < 22:  # smaller than an empty zip central directory
        partial.unlink(missing_ok=True)
        raise FetchError("downloaded export is too small to be a ZIP")
    with partial.open("rb") as handle:
        if handle.read(2) != b"PK":
            partial.unlink(missing_ok=True)
            raise FetchError("downloaded export is not a ZIP container")
    partial.replace(destination)
    return written


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_zip(path: Path) -> dict:
    """Read the archive directory and the CSV header without extracting."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise FetchError(f"{path.name}: expected one member, found {len(names)}")
        member = archive.getinfo(names[0])
        with archive.open(member) as handle:
            header = handle.readline().decode("utf-8", errors="replace").strip()
    return {
        "csv_name": member.filename,
        "csv_bytes": member.file_size,
        "columns": header.split(",") if header else [],
    }


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def fetch_snapshot(snapshot: str, license_expires: str, tables: tuple[str, ...]) -> Path:
    expiry = date.fromisoformat(license_expires)
    if expiry <= datetime.now(timezone.utc).date():
        raise FetchError("license_expires must be in the future")
    token = _read_key()
    snapshot_dir = RAW_ROOT / snapshot
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = snapshot_dir / "manifest.json"

    manifest = _load_manifest(manifest_path)
    manifest.setdefault("schema_version", 1)
    manifest.setdefault("provider", "Sharadar via Nasdaq Data Link")
    manifest.setdefault("package", "SFA")
    manifest.setdefault("api_base", BASE)
    manifest.setdefault("snapshot", snapshot)
    manifest.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    manifest["license_expires"] = expiry.isoformat()
    manifest["expiry_action"] = "Stop use and delete supplied Data when the order expires"
    entries = manifest.setdefault("tables", {})

    for table in tables:
        destination = snapshot_dir / f"{table}.zip"
        recorded = entries.get(table)
        if (
            recorded
            and recorded.get("status") == "COMPLETE"
            and destination.exists()
            and destination.stat().st_size == recorded.get("bytes")
            # Same-length in-place corruption resumes as complete if only the
            # size is checked, so the recorded digest is the real test.
            and _sha256(destination) == recorded.get("sha256")
        ):
            _log(f"{table}: already complete, skipping")
            continue

        _log(f"{table}: requesting export")
        file_info = _wait_for_link(table, token)
        link = file_info["link"]
        _log(f"{table}: downloading from {urllib.parse.urlparse(link).hostname}")
        started = time.monotonic()
        written = _download(link, destination)
        elapsed = time.monotonic() - started
        _log(f"{table}: {written / 1e6:.1f} MB in {elapsed:.0f}s, hashing")

        entry = {
            "table": table,
            "zip_name": destination.name,
            "bytes": written,
            "sha256": _sha256(destination),
            "data_snapshot_time": file_info.get("data_snapshot_time"),
            "link_host": urllib.parse.urlparse(link).hostname,
            "requested_url": _export_url(table),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETE",
        }
        entry.update(_describe_zip(destination))
        entries[table] = entry
        _atomic_write_json(manifest_path, manifest)
        _log(f"{table}: recorded {entry['csv_bytes'] / 1e6:.1f} MB CSV, "
             f"{len(entry['columns'])} columns")

    if set(TABLES) <= {t for t, e in entries.items() if e.get("status") == "COMPLETE"}:
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def purge_snapshot(snapshot: str) -> Path:
    """Delete this script's snapshot and certify it.

    The audited module's `purge` reads a v2 manifest with storage roots and
    cannot touch what this script wrote, so the licence-expiry deletion has to
    live here.
    """
    snapshot_dir = RAW_ROOT / snapshot
    if not snapshot_dir.is_dir():
        raise FetchError(f"no snapshot directory {snapshot_dir}")
    manifest_path = snapshot_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if manifest.get("schema_version") != 1:
        raise FetchError(
            f"{manifest_path} is not a v1 snapshot; use the module's purge instead"
        )
    inventory = []
    for path in sorted(snapshot_dir.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise FetchError(f"refusing to purge unexpected entry {path}")
        inventory.append({"name": path.name, "bytes": path.stat().st_size})
    certificate = {
        "snapshot": snapshot,
        "schema_version": manifest.get("schema_version"),
        "license_expires": manifest.get("license_expires"),
        "purged_at_utc": datetime.now(timezone.utc).isoformat(),
        "deleted": inventory,
    }
    for path in sorted(snapshot_dir.iterdir()):
        path.unlink()
    snapshot_dir.rmdir()
    certificate_dir = Path(__file__).resolve().parent.parent / "reports" / "data_purge"
    certificate_dir.mkdir(parents=True, exist_ok=True)
    certificate_path = certificate_dir / f"{snapshot}.v1.purge.json"
    _atomic_write_json(certificate_path, certificate)
    _log(f"purged {len(inventory)} files; certificate at {certificate_path}")
    return certificate_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--license-expires",
                        help="subscription end date, YYYY-MM-DD (required to fetch)")
    parser.add_argument("--purge", action="store_true",
                        help="delete the snapshot and write a purge certificate")
    parser.add_argument("--snapshot",
                        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--tables", nargs="+", default=list(TABLES))
    args = parser.parse_args()

    if args.purge:
        print(purge_snapshot(args.snapshot))
        return

    unknown = sorted(set(args.tables) - set(TABLES))
    if unknown:
        raise SystemExit(f"unknown tables: {unknown}")
    if not args.license_expires:
        raise SystemExit("--license-expires is required to fetch")

    manifest = fetch_snapshot(args.snapshot, args.license_expires, tuple(args.tables))
    _log(f"manifest written to {manifest}")
    print(manifest)


if __name__ == "__main__":
    os.umask(0o077)
    main()
