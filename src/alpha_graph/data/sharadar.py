"""Staged Sharadar bulk downloads with licensing and provenance guards.

The module intentionally stops at a schema-validated vendor snapshot. It does
not replace ``market_data.parquet`` and does not claim that ``permaticker`` is
a security-level identifier. Identity resolution and coverage gates live in
``alpha_graph.data.sharadar_qa``.

Usage before purchase::

    python -m alpha_graph.data.sharadar plan --profile validation

Usage after purchase::

    python -m alpha_graph.data.sharadar fetch --profile validation \
        --package SFA --license-expires 2026-08-15
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import re
import shutil
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, DATA_DIR, PROJECT_ROOT
from alpha_graph.data.pit_universe import MEMBERSHIP_CSV, RENAME_MAP

BULK_API = "https://data.nasdaq.com/api/v1/bulkdownloads/SHARADAR"
RAW_ROOT = DATA_DIR / "raw" / "sharadar"
STAGED_ROOT = CACHE_DIR / "sharadar"
DEFAULT_START = "2009-01-01"
DEFAULT_BATCH_SIZE = 100
INITIAL_REQUEST_LIMIT_PER_TABLE = 25
TERMS_URL = "https://data.nasdaq.com/terms"
ALLOWED_DOWNLOAD_HOST = "data.nasdaq.com"
ALLOWED_DOWNLOAD_PATH_PREFIX = "/api/v1/bulkdownloads/file/"
ALLOWED_API_PATH_PREFIX = "/api/v1/bulkdownloads/SHARADAR/"
SNAPSHOT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
PACKAGE_TABLES = {
    "SEP": frozenset({"TICKERS", "ACTIONS", "SEP"}),
    "SFA": frozenset(),
}


class SharadarError(RuntimeError):
    """Base error for fail-closed vendor ingestion."""


class SharadarAPIError(SharadarError):
    """Nasdaq bulk API returned a terminal or malformed response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass(frozen=True)
class TableSpec:
    code: str
    primary_key: tuple[str, ...]
    required_columns: frozenset[str]
    date_columns: tuple[str, ...]
    start_filter: str | None
    scope_by_ticker: bool
    nullable_primary_key: frozenset[str] = frozenset()
    allow_empty_export: bool = False


TABLE_SPECS: dict[str, TableSpec] = {
    "TICKERS": TableSpec(
        "TICKERS",
        ("table", "permaticker", "ticker"),
        frozenset({"table", "permaticker", "ticker", "name", "firstpricedate",
                   "lastpricedate"}),
        ("firstpricedate", "lastpricedate", "lastupdated"),
        None,
        False,
    ),
    "SP500": TableSpec(
        "SP500",
        ("date", "ticker", "action"),
        frozenset({"date", "ticker", "action"}),
        ("date",),
        "date",
        False,
    ),
    "ACTIONS": TableSpec(
        "ACTIONS",
        ("date", "ticker", "name", "action", "contraname", "contraticker"),
        frozenset({"date", "ticker", "name", "action", "contraname", "contraticker"}),
        ("date",),
        "date",
        True,
        frozenset({"contraname", "contraticker"}),
        True,
    ),
    "SEP": TableSpec(
        "SEP",
        ("ticker", "date"),
        frozenset({"ticker", "date", "open", "high", "low", "close", "volume",
                   "closeadj", "closeunadj", "dividends", "lastupdated"}),
        ("date", "lastupdated"),
        "date",
        True,
    ),
    "SF1": TableSpec(
        "SF1",
        ("ticker", "dimension", "datekey", "reportperiod"),
        frozenset({"ticker", "dimension", "calendardate", "datekey", "reportperiod",
                   "lastupdated"}),
        ("calendardate", "datekey", "reportperiod", "lastupdated"),
        "datekey",
        True,
    ),
    "SF2": TableSpec(
        "SF2",
        ("ticker", "ownername", "filingdate", "formtype", "rownum"),
        frozenset({"ticker", "ownername", "filingdate", "formtype", "rownum"}),
        ("filingdate", "transactiondate", "lastupdated"),
        "filingdate",
        True,
    ),
    "SF3": TableSpec(
        "SF3",
        ("ticker", "investorname", "calendardate", "securitytype"),
        frozenset({"ticker", "investorname", "calendardate", "securitytype"}),
        ("calendardate", "lastupdated"),
        "calendardate",
        True,
    ),
    "EVENTS": TableSpec(
        "EVENTS",
        ("ticker", "date"),
        frozenset({"ticker", "date"}),
        ("date",),
        "date",
        True,
    ),
    "DAILY": TableSpec(
        "DAILY",
        ("ticker", "date"),
        frozenset({"ticker", "date"}),
        ("date", "lastupdated"),
        "date",
        True,
    ),
}

PROFILES: dict[str, tuple[str, ...]] = {
    "reference": ("TICKERS", "SP500"),
    "validation": ("TICKERS", "SP500", "ACTIONS", "SEP"),
    "research": (
        "TICKERS", "SP500", "ACTIONS", "SEP", "SF1", "SF2", "SF3", "EVENTS", "DAILY"
    ),
}

PACKAGE_TABLES["SFA"] = frozenset(TABLE_SPECS)


@dataclass(frozen=True)
class DownloadPart:
    table: str
    index: int
    filters: tuple[tuple[str, str], ...]

    @property
    def digest(self) -> str:
        raw = json.dumps([self.table, self.filters], separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    @property
    def stem(self) -> str:
        return f"{self.table.lower()}-{self.index:03d}-{self.digest}"


JsonGetter = Callable[[str, str, float], dict]
FileDownloader = Callable[[str, str, Path, float], int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_json(url: str, token: str, timeout: float) -> dict:
    _validate_api_url(url)
    req = Request(url, headers={"X-Api-Token": token, "Accept": "application/json"})
    try:
        opener = build_opener(_NasdaqAPIRedirectHandler())
        with opener.open(req, timeout=timeout) as response:
            _validate_api_url(response.geturl())
            payload = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        retry_after_raw = exc.headers.get("Retry-After") if exc.headers else None
        try:
            retry_after = float(retry_after_raw) if retry_after_raw else None
        except ValueError:
            retry_after = None
        raise SharadarAPIError(
            f"bulk API HTTP {exc.code}: {body}",
            status_code=exc.code,
            retry_after=retry_after,
        ) from exc
    except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
        detail = getattr(exc, "reason", str(exc))
        raise SharadarAPIError(f"bulk API transport error: {detail}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SharadarAPIError("bulk API returned invalid JSON") from exc


def _validate_nasdaq_url(url: str, path_prefix: str, subject: str) -> None:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise SharadarAPIError(f"{subject} URL has an invalid port") from exc
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname != ALLOWED_DOWNLOAD_HOST
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith(path_prefix)
    ):
        raise SharadarAPIError(
            f"{subject} URL is outside the approved Nasdaq HTTPS origin"
        )


def _validate_api_url(url: str) -> None:
    _validate_nasdaq_url(url, ALLOWED_API_PATH_PREFIX, "bulk API")


def _validate_bulk_file_url(url: str) -> None:
    _validate_nasdaq_url(url, ALLOWED_DOWNLOAD_PATH_PREFIX, "bulk file")


class _NasdaqAPIRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_api_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _NasdaqRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_bulk_file_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_file(url: str, token: str, destination: Path, timeout: float) -> int:
    _validate_bulk_file_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    req = Request(url, headers={"X-Api-Token": token})
    written = 0
    try:
        opener = build_opener(_NasdaqRedirectHandler())
        with opener.open(req, timeout=timeout) as response, tmp.open("wb") as fh:
            _validate_bulk_file_url(response.geturl())
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
    except HTTPError as exc:
        retry_after_raw = exc.headers.get("Retry-After") if exc.headers else None
        try:
            retry_after = float(retry_after_raw) if retry_after_raw else None
        except ValueError:
            retry_after = None
        raise SharadarAPIError(
            f"bulk file download HTTP {exc.code} for {urlparse(url).path}",
            status_code=exc.code,
            retry_after=retry_after,
        ) from exc
    except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
        raise SharadarAPIError(f"bulk file download failed for {urlparse(url).path}") from exc
    tmp.replace(destination)
    return written


def build_bulk_url(part: DownloadPart) -> str:
    query = urlencode(list(part.filters), doseq=True)
    return f"{BULK_API}/{part.table}" + (f"?{query}" if query else "")


class NasdaqBulkClient:
    """Small client for the current async Parquet bulk-download API."""

    def __init__(
        self,
        api_key: str,
        *,
        poll_seconds: float = 30.0,
        timeout_seconds: float = 900.0,
        request_timeout: float = 120.0,
        json_getter: JsonGetter = _get_json,
        file_downloader: FileDownloader = _download_file,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 4,
        retry_base_seconds: float = 2.0,
    ) -> None:
        if not api_key:
            raise ValueError("NASDAQ_DATA_LINK_API_KEY is required")
        self._api_key = api_key
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.request_timeout = request_timeout
        self._json_getter = json_getter
        self._file_downloader = file_downloader
        self._sleep = sleep
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def _with_retries(self, operation: Callable[[], object], subject: str) -> object:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except SharadarAPIError as exc:
                retryable = exc.status_code is None or exc.status_code == 429 or (
                    500 <= exc.status_code <= 599
                )
                if not retryable or attempt >= self.max_retries:
                    raise
                delay = exc.retry_after
                if delay is None:
                    delay = self.retry_base_seconds * (2 ** attempt)
                logger.warning(
                    "Sharadar {} transient failure; retry {}/{} in {:.1f}s",
                    subject,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("retry loop did not return or raise")

    def wait_for_export(self, part: DownloadPart) -> dict:
        url = build_bulk_url(part)
        started = time.monotonic()
        while True:
            payload = self._with_retries(
                lambda: self._json_getter(url, self._api_key, self.request_timeout),
                f"{part.table} job",
            )
            if not isinstance(payload, dict):
                raise SharadarAPIError("bulk API response is not an object")
            job = payload.get("bulk_download")
            if not isinstance(job, dict):
                raise SharadarAPIError("bulk API response has no bulk_download object")
            status = str(job.get("status", "")).upper()
            if status == "SUCCEEDED":
                files = job.get("files")
                if not isinstance(files, list):
                    raise SharadarAPIError(f"{part.table} export has no files list")
                if not files and not TABLE_SPECS[part.table].allow_empty_export:
                    raise SharadarAPIError(
                        f"{part.table} export is unexpectedly empty"
                    )
                return job
            if status in {"PENDING", "RUNNING", "RATE_LIMITED"}:
                if time.monotonic() - started >= self.timeout_seconds:
                    raise SharadarAPIError(f"{part.table} export timed out in status {status}")
                self._sleep(self.poll_seconds)
                continue
            rate_until = job.get("rate_limited_until")
            detail = job.get("errors") or rate_until or "no error detail"
            raise SharadarAPIError(f"{part.table} export status {status or 'UNKNOWN'}: {detail}")

    def download_files(self, job: dict, destination: Path, stem: str) -> list[Path]:
        if destination.is_symlink():
            raise SharadarAPIError("bulk destination must not be a symlink")
        destination.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for idx, item in enumerate(job.get("files", [])):
            if not isinstance(item, dict):
                raise SharadarAPIError("bulk file entry is not an object")
            url = item.get("url")
            if not url:
                raise SharadarAPIError("bulk file entry has no URL")
            _validate_bulk_file_url(url)
            suffix = Path(urlparse(url).path).suffix.lower()
            if suffix != ".parquet":
                raise SharadarAPIError(f"unexpected bulk file type {suffix!r}")
            path = destination / f"{stem}-file-{idx:03d}.parquet"
            expected = item.get("size")

            def download_and_validate() -> object:
                path.unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".part").unlink(missing_ok=True)
                written = self._file_downloader(
                    url, self._api_key, path, self.request_timeout
                )
                if not isinstance(written, int):
                    raise SharadarAPIError(
                        f"file downloader returned invalid size for {path.name}"
                    )
                if expected not in (None, 0) and int(expected) != written:
                    raise SharadarAPIError(
                        f"download size mismatch for {path.name}: {written} != {expected}"
                    )
                if written < 8:
                    raise SharadarAPIError(f"{path.name} is too small to be Parquet")
                try:
                    with path.open("rb") as fh:
                        head = fh.read(4)
                        fh.seek(-4, os.SEEK_END)
                        tail = fh.read(4)
                except OSError as exc:
                    raise SharadarAPIError(
                        f"could not validate downloaded file {path.name}"
                    ) from exc
                if head != b"PAR1" or tail != b"PAR1":
                    raise SharadarAPIError(
                        f"{path.name} is not a valid Parquet container"
                    )
                return written

            self._with_retries(download_and_validate, f"file {idx} for {stem}")
            paths.append(path)
        return paths


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def load_scope_tickers(
    membership_csv: Path = MEMBERSHIP_CSV,
    *,
    start: str = DEFAULT_START,
    panel_path: Path | None = CACHE_DIR / "market_data.parquet",
) -> list[str]:
    """Ticker spellings needed to request the historical S&P research panel."""
    frame = pd.read_csv(membership_csv, usecols=["date", "tickers"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"] >= pd.Timestamp(start)]
    symbols: set[str] = set()
    for raw in frame["tickers"]:
        symbols.update(t.strip().upper() for t in raw.split(",") if t.strip())
    symbols.update(new.upper() for new, _ in RENAME_MAP.values())
    if panel_path is not None and panel_path.exists():
        panel = pd.read_parquet(panel_path, columns=["ticker"])
        symbols.update(str(t).upper() for t in panel["ticker"].dropna().unique())

    # Nasdaq and the independent source may use dot versus dash class syntax.
    punctuated = set(symbols)
    for ticker in list(symbols):
        if "." in ticker:
            punctuated.add(ticker.replace(".", "-"))
        if "-" in ticker:
            punctuated.add(ticker.replace("-", "."))
    return sorted(punctuated)


def build_download_plan(
    tables: Iterable[str],
    *,
    tickers: Iterable[str] = (),
    start: str = DEFAULT_START,
    end: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[DownloadPart]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    pd.Timestamp(start)
    if end is not None and pd.Timestamp(end) < pd.Timestamp(start):
        raise ValueError("end must be on or after start")
    scope = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    plan: list[DownloadPart] = []
    for raw_table in tables:
        table = raw_table.upper()
        if table not in TABLE_SPECS:
            raise ValueError(f"unsupported Sharadar table {table!r}")
        spec = TABLE_SPECS[table]
        common: list[tuple[str, str]] = []
        if spec.start_filter:
            common.append((f"{spec.start_filter}.gte", str(pd.Timestamp(start).date())))
            # SP500 is reconstructed backward from current rows, so events
            # after the local audit end must remain in the export.
            if end is not None and table != "SP500":
                common.append((f"{spec.start_filter}.lte", str(pd.Timestamp(end).date())))
        if table == "TICKERS":
            common.append(("table.eq", "SEP"))
        if spec.scope_by_ticker:
            if not scope:
                raise ValueError(f"{table} requires a non-empty ticker scope")
            for index, batch in enumerate(_chunks(scope, batch_size)):
                filters = common + [("ticker.in[]", ticker) for ticker in batch]
                plan.append(DownloadPart(table, index, tuple(filters)))
        else:
            plan.append(DownloadPart(table, 0, tuple(common)))

    counts = Counter(part.table for part in plan)
    excess = {table: count for table, count in counts.items()
              if count > INITIAL_REQUEST_LIMIT_PER_TABLE}
    if excess:
        raise ValueError(
            f"plan exceeds the initial {INITIAL_REQUEST_LIMIT_PER_TABLE}-request/table "
            f"limit: {excess}; increase batch_size"
        )
    return plan


def plan_summary(plan: list[DownloadPart]) -> dict:
    return {
        "schema_version": 1,
        "api": BULK_API,
        "request_count": len(plan),
        "requests_by_table": dict(sorted(Counter(p.table for p in plan).items())),
        "parts": [asdict(part) | {"digest": part.digest, "url": build_bulk_url(part)}
                  for part in plan],
        "authentication": "X-Api-Token header; token is never stored in this plan",
    }


def validate_license_expiry(expires: str, *, today: date | None = None) -> str:
    expiry = pd.Timestamp(expires).date()
    today = today or datetime.now(timezone.utc).date()
    if expiry < today:
        raise SharadarError(f"license expired on {expiry}; vendor data use must stop")
    return str(expiry)


def _validate_and_stage(raw_path: Path, staged_path: Path, spec: TableSpec) -> dict:
    frame = pd.read_parquet(raw_path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = sorted(spec.required_columns - set(frame.columns))
    if missing:
        raise SharadarError(f"{spec.code} missing required columns: {missing}")
    for column in spec.date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    nonnullable_key = [
        column for column in spec.primary_key if column not in spec.nullable_primary_key
    ]
    if frame[nonnullable_key].isna().any().any():
        raise SharadarError(f"{spec.code} has null primary-key values")
    if frame.duplicated(list(spec.primary_key)).any():
        raise SharadarError(f"{spec.code} has duplicate primary keys within {raw_path.name}")

    if spec.code == "SEP" and not frame.empty:
        numeric = [
            "open", "high", "low", "close", "closeadj", "closeunadj",
            "volume", "dividends",
        ]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        invalid_numeric = frame[numeric].isna() | ~np.isfinite(frame[numeric])
        if invalid_numeric.any().any():
            bad_columns = sorted(invalid_numeric.any()[invalid_numeric.any()].index)
            raise SharadarError(
                f"SEP has null or non-finite numeric fields {bad_columns} in {raw_path.name}"
            )
        positive = ["open", "high", "low", "close", "closeadj", "closeunadj"]
        for column in positive:
            if (frame[column].dropna() <= 0).any():
                raise SharadarError(f"SEP has non-positive {column} in {raw_path.name}")
        ohlc = frame[["open", "high", "low", "close"]].dropna()
        bad_high = ohlc["high"] < ohlc[["open", "low", "close"]].max(axis=1)
        bad_low = ohlc["low"] > ohlc[["open", "high", "close"]].min(axis=1)
        if bad_high.any() or bad_low.any():
            raise SharadarError(f"SEP has OHLC ordering violations in {raw_path.name}")
        if (frame["volume"].dropna() < 0).any():
            raise SharadarError(f"SEP has negative volume in {raw_path.name}")

    frame = frame.sort_values(list(spec.primary_key)).reset_index(drop=True)
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = staged_path.with_suffix(staged_path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    tmp.replace(staged_path)
    first_date = spec.date_columns[0] if spec.date_columns else None
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "first_date": (str(frame[first_date].min()) if first_date and len(frame) else None),
        "last_date": (str(frame[first_date].max()) if first_date and len(frame) else None),
        "sha256": sha256_file(staged_path),
    }


def _record_complete(record: dict, raw_root: Path, staged_root: Path) -> bool:
    files = record.get("files", [])
    if record.get("status") != "SUCCEEDED":
        return False
    if record.get("empty_export"):
        return files == []
    if not files:
        return False
    for item in files:
        raw = raw_root / item["raw_path"]
        staged = staged_root / item["staged_path"]
        if not raw.exists() or not staged.exists():
            return False
        if raw.stat().st_size != item.get("raw_size"):
            return False
        if sha256_file(raw) != item["raw_sha256"]:
            return False
        if sha256_file(staged) != item["staged_sha256"]:
            return False
    return True


def validate_snapshot_id(snapshot: str) -> str:
    if not SNAPSHOT_RE.fullmatch(snapshot) or snapshot in {".", ".."}:
        raise ValueError(
            "snapshot must be a single safe component containing only letters, "
            "digits, dot, underscore, or dash"
        )
    return snapshot


def _snapshot_path(root: Path, snapshot: str) -> Path:
    root_resolved = root.resolve()
    lexical = root_resolved / validate_snapshot_id(snapshot)
    if lexical.is_symlink():
        raise SharadarError("snapshot path must not be a symlink")
    candidate = lexical.resolve()
    if candidate != lexical or candidate.parent != root_resolved:
        raise SharadarError("snapshot path escapes its configured root")
    return candidate


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise SharadarError(f"symlinked snapshot path is prohibited: {root}")
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise SharadarError(f"symlinked snapshot entry is prohibited: {path}")


def _safe_table_dir(snapshot_dir: Path, table: str) -> Path:
    if table not in TABLE_SPECS:
        raise SharadarError(f"unsupported snapshot table {table}")
    path = snapshot_dir / table
    if path.is_symlink():
        raise SharadarError(f"table directory must not be a symlink: {path}")
    if path.exists() and path.resolve() != path:
        raise SharadarError(f"table directory escapes the snapshot: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if path.resolve().parent != snapshot_dir.resolve():
        raise SharadarError(f"table directory escapes the snapshot: {path}")
    return path


@contextmanager
def snapshot_lock(
    raw_root: Path,
    snapshot: str,
    *,
    additional_roots: Iterable[Path] = (),
):
    """Acquire crash-safe OS locks for every storage root in stable order."""
    validate_snapshot_id(snapshot)
    roots = sorted({Path(raw_root).resolve(), *(
        Path(root).resolve() for root in additional_roots
    )}, key=str)
    handles = []
    try:
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
            lock_path = root / f".{snapshot}.lock"
            handle = lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise SharadarError(
                    f"snapshot {snapshot} is locked by another process"
                ) from exc
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "created_at_utc": _utc_now()}))
            handle.flush()
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _planned_part(part: DownloadPart) -> dict:
    return {
        "stem": part.stem,
        "table": part.table,
        "index": part.index,
        "filters": [list(value) for value in part.filters],
        "digest": part.digest,
    }


def _license_payload(manifest: dict) -> dict:
    """Derive the snapshot license record from its manifest fields."""
    return {
        "provider": manifest.get("provider", "Sharadar via Nasdaq Data Link"),
        "package": manifest["package"],
        "snapshot": manifest["snapshot"],
        "license_expires": manifest["license_expires"],
        "terms_url": manifest.get("terms_url", TERMS_URL),
        "expiry_action": (
            "Stop use and delete supplied Data unless written terms say otherwise"
        ),
    }


def verify_snapshot_manifest(
    manifest_path: Path,
    *,
    raw_snapshot: Path,
    staged_snapshot: Path,
    enforce_license: bool = True,
) -> dict:
    """Verify a COMPLETE manifest, exact inventories, and every file hash."""
    if not manifest_path.exists():
        raise SharadarError(f"missing snapshot manifest {manifest_path}")
    _reject_symlinks(raw_snapshot)
    _reject_symlinks(staged_snapshot)
    if raw_snapshot.resolve() == staged_snapshot.resolve():
        raise SharadarError("raw and staged snapshot paths must be distinct")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise SharadarError("snapshot manifest is not schema version 2")
    if manifest.get("status") != "COMPLETE":
        raise SharadarError("snapshot manifest is not COMPLETE")
    if manifest.get("snapshot") != raw_snapshot.name:
        raise SharadarError("manifest snapshot ID does not match its directory")
    package = manifest.get("package")
    if package not in PACKAGE_TABLES:
        raise SharadarError("manifest has an unsupported package")
    if enforce_license:
        validate_license_expiry(manifest["license_expires"])
    storage_roots = manifest.get("storage_roots", {})
    if (
        storage_roots.get("raw") != str(raw_snapshot.parent.resolve())
        or storage_roots.get("staged") != str(staged_snapshot.parent.resolve())
    ):
        raise SharadarError("manifest storage roots do not match snapshot locations")
    planned = manifest.get("planned_parts")
    parts = manifest.get("parts")
    if not isinstance(planned, list) or not isinstance(parts, list):
        raise SharadarError("snapshot manifest has no planned/complete part inventory")
    planned_stems = [item.get("stem") for item in planned]
    completed_stems = [item.get("stem") for item in parts]
    if (
        len(planned_stems) != len(set(planned_stems))
        or len(completed_stems) != len(set(completed_stems))
        or set(planned_stems) != set(completed_stems)
    ):
        raise SharadarError("planned and completed manifest parts differ")
    reconstructed_plan = []
    for item in planned:
        filters = tuple(tuple(value) for value in item.get("filters", []))
        part = DownloadPart(str(item.get("table", "")), item.get("index"), filters)
        if item.get("stem") != part.stem or item.get("digest") != part.digest:
            raise SharadarError("planned-part digest or stem is inconsistent")
        reconstructed_plan.append(asdict(part))
    plan_digest = hashlib.sha256(
        json.dumps(reconstructed_plan, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if manifest.get("plan_sha256") != plan_digest:
        raise SharadarError("manifest plan digest does not match planned parts")
    tables = {str(item.get("table", "")) for item in planned}
    if not tables <= PACKAGE_TABLES[package]:
        raise SharadarError(f"manifest package {package} does not entitle {sorted(tables)}")

    license_path = raw_snapshot / "license.json"
    if not license_path.exists():
        raise SharadarError("snapshot has no license metadata")
    license_record = json.loads(license_path.read_text(encoding="utf-8"))
    for manifest_key, license_key in (
        ("snapshot", "snapshot"),
        ("package", "package"),
        ("license_expires", "license_expires"),
    ):
        if manifest.get(manifest_key) != license_record.get(license_key):
            raise SharadarError(f"license metadata mismatch for {manifest_key}")
    completed = {item["stem"]: item for item in parts}
    raw_base = raw_snapshot.resolve()
    staged_base = staged_snapshot.resolve()
    for item in planned:
        record = completed[item["stem"]]
        for key in ("table", "index", "filters"):
            if record.get(key) != item.get(key):
                raise SharadarError(f"manifest query mismatch for {item['stem']}")
        for file in record.get("files", []):
            raw_path = (raw_snapshot / file["raw_path"]).resolve()
            staged_path = (staged_snapshot / file["staged_path"]).resolve()
            if (
                not raw_path.is_relative_to(raw_base)
                or not staged_path.is_relative_to(staged_base)
            ):
                raise SharadarError("manifest file path escapes its snapshot root")
        if not _record_complete(record, raw_snapshot, staged_snapshot):
            raise SharadarError(f"manifest checksum verification failed for {item['stem']}")

    recorded_raw = {
        str((raw_snapshot / file["raw_path"]).resolve())
        for part in parts for file in part.get("files", [])
    }
    recorded_staged = {
        str((staged_snapshot / file["staged_path"]).resolve())
        for part in parts for file in part.get("files", [])
    }
    allowed_raw = recorded_raw | {str(manifest_path.resolve()), str(license_path.resolve())}
    actual_raw = {
        str(path.resolve()) for path in raw_snapshot.rglob("*") if path.is_file()
    }
    actual_staged = {
        str(path.resolve()) for path in staged_snapshot.rglob("*") if path.is_file()
    }
    if actual_raw != allowed_raw or actual_staged != recorded_staged:
        raise SharadarError("snapshot contains unmanifested or missing files")
    return manifest


def _fetch_snapshot_unlocked(
    client: NasdaqBulkClient,
    plan: list[DownloadPart],
    *,
    snapshot: str,
    package: str,
    license_expires: str,
    raw_root: Path = RAW_ROOT,
    staged_root: Path = STAGED_ROOT,
    qa_root: Path = DATA_DIR / "qa" / "sharadar",
) -> Path:
    """Fetch and stage a resumable immutable snapshot; return manifest path."""
    validate_snapshot_id(snapshot)
    package = package.upper()
    if package not in PACKAGE_TABLES:
        raise ValueError("package must be SEP or SFA")
    unentitled = sorted({part.table for part in plan} - PACKAGE_TABLES[package])
    if unentitled:
        raise ValueError(f"{package} does not entitle tables: {unentitled}")
    if not plan:
        raise ValueError("download plan must not be empty")
    if len({part.stem for part in plan}) != len(plan):
        raise ValueError("download plan contains duplicate part stems")
    expiry = validate_license_expiry(license_expires)
    raw_snapshot = _snapshot_path(raw_root, snapshot)
    staged_snapshot = _snapshot_path(staged_root, snapshot)
    if raw_snapshot == staged_snapshot:
        raise SharadarError("raw and staged snapshot paths must be distinct")
    _reject_symlinks(raw_snapshot)
    _reject_symlinks(staged_snapshot)
    manifest_path = raw_snapshot / "manifest.json"
    license_path = raw_snapshot / "license.json"
    plan_digest = hashlib.sha256(
        json.dumps([asdict(p) for p in plan], sort_keys=True).encode("utf-8")
    ).hexdigest()

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 2:
            raise SharadarError("existing snapshot uses an unsupported manifest schema")
        if (
            manifest.get("package") != package
            or manifest.get("license_expires") != expiry
        ):
            raise SharadarError("snapshot already exists with different license metadata")
        expected_roots = {
            "raw": str(raw_root.resolve()),
            "staged": str(staged_root.resolve()),
            "qa": str(qa_root.resolve()),
        }
        if manifest.get("storage_roots") != expected_roots:
            raise SharadarError("snapshot already exists with different storage roots")
        if manifest.get("plan_sha256") != plan_digest:
            raise SharadarError(
                "snapshot plan is immutable; use a new snapshot ID for a different plan"
            )
        if manifest.get("planned_parts") != [_planned_part(part) for part in plan]:
            raise SharadarError("snapshot planned-part inventory does not match the plan")
        if license_path.exists():
            license_record = json.loads(license_path.read_text(encoding="utf-8"))
            for manifest_key, license_key in (
                ("snapshot", "snapshot"),
                ("package", "package"),
                ("license_expires", "license_expires"),
            ):
                if manifest.get(manifest_key) != license_record.get(license_key):
                    raise SharadarError(
                        f"snapshot license metadata mismatch for {manifest_key}"
                    )
        else:
            # A crash between the manifest and license writes, or a stray
            # deletion, must not leave the snapshot unresumable.
            _atomic_json(license_path, _license_payload(manifest))
        if manifest.get("status") == "COMPLETE":
            verify_snapshot_manifest(
                manifest_path,
                raw_snapshot=raw_snapshot,
                staged_snapshot=staged_snapshot,
            )
            return manifest_path
    else:
        manifest = {
            "schema_version": 2,
            "provider": "Sharadar via Nasdaq Data Link",
            "snapshot": snapshot,
            "package": package,
            "license_expires": expiry,
            "storage_roots": {
                "raw": str(raw_root.resolve()),
                "staged": str(staged_root.resolve()),
                "qa": str(qa_root.resolve()),
            },
            "plan_sha256": plan_digest,
            "planned_parts": [_planned_part(part) for part in plan],
            "created_at_utc": _utc_now(),
            "terms_url": TERMS_URL,
            "status": "IN_PROGRESS",
            "parts": [],
        }
        _atomic_json(manifest_path, manifest)
        _atomic_json(license_path, _license_payload(manifest))

    records = {item["stem"]: item for item in manifest["parts"]}
    for part in plan:
        prior = records.get(part.stem)
        if prior and _record_complete(prior, raw_snapshot, staged_snapshot):
            logger.info("Sharadar {} already complete; checksum verified", part.stem)
            continue
        logger.info("Requesting Sharadar {}", part.stem)
        raw_dir = _safe_table_dir(raw_snapshot, part.table)
        staged_dir = _safe_table_dir(staged_snapshot, part.table)
        for orphan in list(raw_dir.glob(f"{part.stem}-file-*.parquet*")):
            orphan.unlink()
        for orphan in list(staged_dir.glob(f"{part.stem}-file-*.parquet*")):
            orphan.unlink()
        job = client.wait_for_export(part)
        raw_files = client.download_files(job, raw_dir, part.stem)
        file_records = []
        for raw_file in raw_files:
            staged_file = staged_snapshot / part.table / raw_file.name
            stage = _validate_and_stage(raw_file, staged_file, TABLE_SPECS[part.table])
            file_records.append({
                "raw_path": str(raw_file.relative_to(raw_snapshot)),
                "raw_size": raw_file.stat().st_size,
                "raw_sha256": sha256_file(raw_file),
                "staged_path": str(staged_file.relative_to(staged_snapshot)),
                "staged_sha256": stage.pop("sha256"),
                **stage,
            })
        record = {
            "stem": part.stem,
            "table": part.table,
            "index": part.index,
            "filters": list(part.filters),
            "requested_at_utc": _utc_now(),
            "status": job.get("status"),
            "empty_export": not raw_files,
            "schema": job.get("schema"),
            "files": file_records,
        }
        manifest["parts"] = [item for item in manifest["parts"]
                             if item["stem"] != part.stem] + [record]
        manifest["parts"].sort(key=lambda item: (item["table"], item["index"]))
        manifest["updated_at_utc"] = _utc_now()
        _atomic_json(manifest_path, manifest)
    planned_stems = {part.stem for part in plan}
    complete_stems = {item["stem"] for item in manifest["parts"]}
    if planned_stems != complete_stems or not all(
        _record_complete(item, raw_snapshot, staged_snapshot)
        for item in manifest["parts"]
    ):
        raise SharadarError("snapshot did not complete every planned part")
    manifest["status"] = "COMPLETE"
    manifest["completed_at_utc"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    verify_snapshot_manifest(
        manifest_path,
        raw_snapshot=raw_snapshot,
        staged_snapshot=staged_snapshot,
    )
    return manifest_path


def fetch_snapshot(
    client: NasdaqBulkClient,
    plan: list[DownloadPart],
    *,
    snapshot: str,
    package: str,
    license_expires: str,
    raw_root: Path = RAW_ROOT,
    staged_root: Path = STAGED_ROOT,
    qa_root: Path = DATA_DIR / "qa" / "sharadar",
) -> Path:
    """Serialize all writes for one immutable snapshot."""
    with snapshot_lock(
        raw_root,
        snapshot,
        additional_roots=(staged_root, qa_root),
    ):
        return _fetch_snapshot_unlocked(
            client,
            plan,
            snapshot=snapshot,
            package=package,
            license_expires=license_expires,
            raw_root=raw_root,
            staged_root=staged_root,
            qa_root=qa_root,
        )


def _profile_plan(args: argparse.Namespace) -> list[DownloadPart]:
    tables = tuple(args.tables) if args.tables else PROFILES[args.profile]
    needs_scope = any(TABLE_SPECS[t.upper()].scope_by_ticker for t in tables)
    tickers = load_scope_tickers(Path(args.membership), start=args.start) if needs_scope else []
    return build_download_plan(
        tables,
        tickers=tickers,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
    )


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=sorted(PROFILES), default="validation")
    parser.add_argument("--tables", nargs="+", choices=sorted(TABLE_SPECS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--membership", default=str(MEMBERSHIP_CSV))


def _purge_snapshot_data_unlocked(
    snapshot: str,
    *,
    raw_root: Path = RAW_ROOT,
    staged_root: Path = STAGED_ROOT,
    qa_root: Path = DATA_DIR / "qa" / "sharadar",
    certificate_root: Path = PROJECT_ROOT / "reports" / "data_purge",
) -> Path:
    """Delete one vendor snapshot through a resumable journal, then certify.

    The journal is written before any deletion and carries the complete
    inventory plus per-root phase markers (QA, then staged, then raw). An
    interrupted purge resumes from the journal even after the raw manifest
    is gone; the certificate is written only after every phase completes.
    """
    raw_snapshot = _snapshot_path(raw_root, snapshot)
    staged_snapshot = _snapshot_path(staged_root, snapshot)
    qa_snapshot = _snapshot_path(qa_root, snapshot)
    roots = (("qa", qa_snapshot), ("staged", staged_snapshot), ("raw", raw_snapshot))
    expected_roots = {
        "raw": str(raw_root.resolve()),
        "staged": str(staged_root.resolve()),
        "qa": str(qa_root.resolve()),
    }
    certificate_root.mkdir(parents=True, exist_ok=True)
    journal_path = certificate_root / f"{snapshot}.journal.json"
    certificate = certificate_root / f"{snapshot}.json"

    journal = None
    if journal_path.exists():
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal.get("schema_version") != 1 or journal.get("snapshot") != snapshot:
            raise SharadarError(f"purge journal does not match snapshot {snapshot}")
        if journal.get("storage_roots") != expected_roots:
            raise SharadarError("purge journal roots do not match the requested roots")
        if journal.get("status") == "COMPLETE":
            if any(root.exists() or root.is_symlink() for _, root in roots):
                # The snapshot name was reused after a completed purge;
                # journal the new deletion from scratch.
                journal = None
            elif certificate.exists():
                return certificate

    if journal is None:
        manifest_path = raw_snapshot / "manifest.json"
        if not manifest_path.exists():
            raise SharadarError(f"cannot purge {snapshot}: manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("storage_roots") != expected_roots:
            raise SharadarError("purge roots do not match the snapshot manifest")
        for _, root in roots:
            _reject_symlinks(root)
        inventory = []
        for kind, root in roots:
            if root.exists():
                inventory.extend({
                    "kind": kind,
                    "relative_path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                } for path in root.rglob("*") if path.is_file())
        journal = {
            "schema_version": 1,
            "snapshot": snapshot,
            "created_at_utc": _utc_now(),
            "storage_roots": expected_roots,
            "manifest_sha256_before_purge": sha256_file(manifest_path),
            "deleted_inventory": inventory,
            "phases": {kind: "PENDING" for kind, _ in roots},
            "status": "IN_PROGRESS",
        }
        _atomic_json(journal_path, journal)

    for kind, root in roots:
        if root.exists() or root.is_symlink():
            _reject_symlinks(root)
            shutil.rmtree(root)
        if journal["phases"].get(kind) != "DELETED":
            journal["phases"][kind] = "DELETED"
            journal["updated_at_utc"] = _utc_now()
            _atomic_json(journal_path, journal)

    remaining = [
        str(root) for _, root in roots if root.exists() or root.is_symlink()
    ]
    if remaining:
        raise SharadarError(f"purge left snapshot artifacts behind: {remaining}")
    inventory = journal["deleted_inventory"]
    _atomic_json(certificate, {
        "schema_version": 1,
        "snapshot": snapshot,
        "purged_at_utc": _utc_now(),
        "manifest_sha256_before_purge": journal["manifest_sha256_before_purge"],
        "deleted_file_count": len(inventory),
        "deleted_bytes": sum(item["bytes"] for item in inventory),
        "deleted_inventory": inventory,
    })
    journal["status"] = "COMPLETE"
    journal["completed_at_utc"] = _utc_now()
    _atomic_json(journal_path, journal)
    return certificate


def purge_snapshot_data(
    snapshot: str,
    *,
    raw_root: Path = RAW_ROOT,
    staged_root: Path = STAGED_ROOT,
    qa_root: Path = DATA_DIR / "qa" / "sharadar",
    certificate_root: Path = PROJECT_ROOT / "reports" / "data_purge",
) -> Path:
    with snapshot_lock(
        raw_root,
        snapshot,
        additional_roots=(staged_root, qa_root),
    ):
        return _purge_snapshot_data_unlocked(
            snapshot,
            raw_root=raw_root,
            staged_root=staged_root,
            qa_root=qa_root,
            certificate_root=certificate_root,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or fetch a staged Sharadar snapshot")
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="print an unauthenticated download plan")
    _add_plan_args(plan_parser)
    fetch_parser = sub.add_parser("fetch", help="execute and stage the authenticated plan")
    _add_plan_args(fetch_parser)
    fetch_parser.add_argument(
        "--snapshot",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    fetch_parser.add_argument("--package", choices=("SFA", "SEP"), default="SFA")
    fetch_parser.add_argument("--license-expires", required=True)
    fetch_parser.add_argument("--api-key-env", default="NASDAQ_DATA_LINK_API_KEY")
    purge_parser = sub.add_parser("purge", help="delete a snapshot after license expiry")
    purge_parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()

    if args.command == "purge":
        print(purge_snapshot_data(args.snapshot))
        return

    plan = _profile_plan(args)
    if args.command == "plan":
        print(json.dumps(plan_summary(plan), indent=2))
        return

    api_key = os.getenv(args.api_key_env, "")
    client = NasdaqBulkClient(api_key)
    manifest = fetch_snapshot(
        client,
        plan,
        snapshot=args.snapshot,
        package=args.package,
        license_expires=args.license_expires,
    )
    print(manifest)


if __name__ == "__main__":
    main()
