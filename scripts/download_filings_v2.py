"""Resumable SEC EDGAR downloader: manifest checkpointing, content-validated
skip (re-fetches corrupt/thin files), retry with backoff. Output format is
identical to alpha_graph.data.filings.

Usage:
    python scripts/download_filings_v2.py --forms 10-K --start-year 2011 --end-year 2026
    python scripts/download_filings_v2.py --tickers MSFT JPM --forms 10-K
    python scripts/download_filings_v2.py --revalidate
    python scripts/download_filings_v2.py --coverage-only
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from loguru import logger

# Reuse the project's extraction + identity so the on-disk format stays identical.
from alpha_graph.config import FILINGS_DIR, cfg
from alpha_graph.data.filings import (
    SEC_REQUEST_DELAY,
    _extract_8k_sections,
    _extract_10k_sections,
    _extract_10q_sections,
    _init_edgar,
)

MANIFEST_PATH = FILINGS_DIR / "_manifest.json"
MIN_SECTION_CHARS = 500   # below this, treat the filing as a failed extraction
RETRIES = 3
BACKOFF_BASE = 1.6        # wait = BACKOFF_BASE ** attempt seconds

_EXTRACTORS = {
    "10-K": _extract_10k_sections,
    "10-Q": _extract_10q_sections,
    "8-K": _extract_8k_sections,
}

# Set by the SIGINT handler so the inner loop can stop cleanly and flush.
_INTERRUPTED = False


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Manifest (atomic load/save)
# --------------------------------------------------------------------------- #

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Manifest unreadable ({e}); starting fresh")
    return {}


def save_manifest(manifest: dict) -> None:
    """Atomic write: tmp file + rename, so a crash mid-write can't corrupt it."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


def _key(ticker: str, form: str) -> str:
    return f"{ticker}|{form}"


# --------------------------------------------------------------------------- #
# On-disk validation
# --------------------------------------------------------------------------- #

def _section_chars(record: dict) -> int:
    return sum(len(v) for v in record.get("sections", {}).values())


def validate_file(path: Path) -> tuple[bool, str]:
    """Return (ok, kind). kind in {'structured', 'fallback', 'thin', 'bad'}."""
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, "bad"
    sections = rec.get("sections", {})
    if not sections:
        return False, "bad"
    if _section_chars(rec) < MIN_SECTION_CHARS:
        return False, "thin"
    kind = "fallback" if set(sections.keys()) == {"full_text"} else "structured"
    return True, kind


def index_existing(ticker_dir: Path, form: str) -> dict[str, Path]:
    """Map accession (no dashes) -> path for already-downloaded files of this form."""
    out: dict[str, Path] = {}
    if not ticker_dir.exists():
        return out
    for f in ticker_dir.glob(f"{form}_*.json"):
        # filename: {form}_{YYYY-MM-DD}_{accession}.json  (form may contain '-')
        acc = f.stem.rsplit("_", 1)[-1]
        out[acc] = f
    return out


# --------------------------------------------------------------------------- #
# Retry wrapper
# --------------------------------------------------------------------------- #

def with_retry(label: str, fn, retries: int = RETRIES):
    last = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - EDGAR raises a zoo of errors
            last = e
            if attempt == retries:
                break
            wait = BACKOFF_BASE ** attempt
            logger.warning(f"{label}: attempt {attempt + 1}/{retries + 1} failed ({e}); retry in {wait:.1f}s")
            time.sleep(wait)
    raise last  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Core: one (ticker, form)
# --------------------------------------------------------------------------- #

def process_ticker_form(
    company, ticker: str, form: str, years: list[int],
    manifest: dict, *, revalidate: bool, stats: dict,
) -> None:
    from edgar import Company  # noqa: F401  (company already constructed)

    key = _key(ticker, form)
    entry = manifest.get(key, {})
    requested = [years[0], years[-1]]

    # Fast skip: completed for a range covering what we want, and not revalidating.
    if not revalidate and entry.get("status") == "complete":
        have = entry.get("years", [9999, 0])
        if have[0] <= requested[0] and have[1] >= requested[1]:
            stats["skipped_complete"] += 1
            return

    ticker_dir = FILINGS_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    existing = index_existing(ticker_dir, form)

    try:
        # amendments=False: /A refilings duplicate a period and, mislabeled as
        # originals, inject spurious near-zero-gap pairs into the text factors.
        filings = with_retry(f"[{ticker}] list {form}",
                             lambda: company.get_filings(form=form, year=years, amendments=False))
    except Exception as e:  # noqa: BLE001
        manifest[key] = {**entry, "ticker": ticker, "form": form, "status": "failed",
                         "error": f"list: {e}", "last_attempt": _now()}
        stats["failed_list"] += 1
        logger.error(f"[{ticker}] {form} listing failed after retries: {e}")
        return

    if filings is None:
        manifest[key] = {"ticker": ticker, "form": form, "status": "complete",
                         "n_files": len(existing), "years": requested,
                         "last_attempt": _now(), "error": None}
        return

    n_total = len(filings)
    n_new = n_fallback = n_failed = 0
    accessions: list[str] = []

    for i in range(n_total):
        if _INTERRUPTED:
            break
        try:
            filing = filings[i]
        except Exception:  # noqa: BLE001
            break

        acc_dashes = filing.accession_number
        acc = acc_dashes.replace("-", "")
        filing_date = str(filing.filing_date)
        out_path = ticker_dir / f"{form}_{filing_date}_{acc}.json"
        accessions.append(acc)

        # Validation-aware skip: only skip if the existing file is genuinely good.
        if acc in existing and out_path.exists():
            ok, kind = validate_file(out_path)
            if ok:
                if kind == "fallback":
                    n_fallback += 1
                continue
            logger.info(f"[{ticker}] re-fetching {out_path.name} (invalid: {kind})")

        # Extract + write
        try:
            sections = with_retry(f"[{ticker}] extract {form} {filing_date}",
                                  lambda f=filing: _EXTRACTORS[form](f))
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            logger.warning(f"[{ticker}] {form} {filing_date} extraction failed: {e}")
            time.sleep(SEC_REQUEST_DELAY)
            continue

        total_chars = sum(len(v) for v in sections.values())
        if not sections or total_chars < MIN_SECTION_CHARS:
            n_failed += 1
            logger.warning(f"[{ticker}] {form} {filing_date} thin extraction ({total_chars} chars); not saved")
            time.sleep(SEC_REQUEST_DELAY)
            continue

        record = {
            "ticker": ticker,
            "company_name": company.name,
            "form_type": form,
            "filing_date": filing_date,
            "accession_number": acc_dashes,
            "sections": sections,
        }
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        n_new += 1
        if set(sections.keys()) == {"full_text"}:
            n_fallback += 1
        logger.info(f"[{ticker}] saved {form} {filing_date} ({total_chars:,} chars)")
        time.sleep(SEC_REQUEST_DELAY)

    status = "complete" if (n_failed == 0 and not _INTERRUPTED) else "partial"
    manifest[key] = {
        "ticker": ticker, "form": form, "status": status,
        "n_files": len(index_existing(ticker_dir, form)),
        "n_listed": n_total, "n_new": n_new, "n_fallback": n_fallback,
        "n_failed": n_failed, "years": requested,
        "last_attempt": _now(),
        "error": None if n_failed == 0 else f"{n_failed} extraction failures",
    }
    stats["new_files"] += n_new
    stats["failed_files"] += n_failed
    if n_new:
        logger.success(f"[{ticker}] {form}: +{n_new} new (total {manifest[key]['n_files']}, "
                       f"{n_fallback} fallback, {n_failed} failed)")


# --------------------------------------------------------------------------- #
# Coverage report
# --------------------------------------------------------------------------- #

def coverage_report(forms: list[str], universe_n: int = 499) -> None:
    print("\n" + "=" * 60)
    print("  PER-YEAR COVERAGE (tickers with >=1 valid filing that year)")
    print("=" * 60)
    for form in forms:
        by_year: dict[int, set] = defaultdict(set)
        for tdir in FILINGS_DIR.iterdir():
            if not tdir.is_dir() or tdir.name.startswith("."):
                continue
            for f in tdir.glob(f"{form}_*.json"):
                year = int(f.stem.split("_")[1][:4])
                by_year[year].add(tdir.name)
        print(f"\n{form}:")
        for y in sorted(by_year):
            n = len(by_year[y])
            print(f"  {y}  {n:4d}  {n / universe_n * 100:5.1f}%  {'#' * int(n / universe_n * 50)}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _install_sigint():
    def handler(signum, frame):  # noqa: ARG001
        global _INTERRUPTED
        _INTERRUPTED = True
        logger.warning("Interrupt received — finishing current filing, then flushing manifest...")
    signal.signal(signal.SIGINT, handler)


def main():
    p = argparse.ArgumentParser(description="Rigorous resumable SEC filing downloader")
    p.add_argument("--tickers", nargs="+", help="Specific tickers (default: full universe)")
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--forms", nargs="+", default=["10-K"], choices=["10-K", "10-Q", "8-K"])
    p.add_argument("--start-year", type=int, default=2011)
    p.add_argument("--end-year", type=int, default=datetime.now().year)
    p.add_argument("--revalidate", action="store_true",
                   help="Re-check on-disk files and re-fetch invalid ones (ignores fast-skip)")
    p.add_argument("--coverage-only", action="store_true", help="Print coverage and exit")
    args = p.parse_args()

    if args.coverage_only:
        coverage_report(args.forms)
        return

    _init_edgar()
    _install_sigint()
    from edgar import Company

    if args.tickers:
        tickers = args.tickers
    else:
        # Default to the exact panel universe (the tickers with price data in
        # market_data), NOT get_sp500_tickers() — which silently caps at
        # cfg.max_tickers=100 via `max_tickers or cfg.max_tickers`.
        import pandas as pd
        from alpha_graph.config import CACHE_DIR
        mkt = pd.read_parquet(CACHE_DIR / "market_data.parquet", columns=["ticker"])
        tickers = sorted(mkt["ticker"].unique().tolist())
        if args.max_tickers:
            tickers = tickers[: args.max_tickers]

    years = list(range(args.start_year, args.end_year + 1))
    manifest = load_manifest()
    stats = defaultdict(int)
    logger.info(f"Downloading forms={args.forms} years={years[0]}-{years[-1]} "
                f"for {len(tickers)} tickers (revalidate={args.revalidate})")

    # form-major: finish the most valuable form across all tickers first
    for form in args.forms:
        if _INTERRUPTED:
            break
        logger.info(f"===== FORM {form} =====")
        for idx, ticker in enumerate(tickers, 1):
            if _INTERRUPTED:
                break
            try:
                company = with_retry(f"[{ticker}] Company()", lambda t=ticker: Company(t))
            except Exception as e:  # noqa: BLE001
                manifest[_key(ticker, form)] = {"ticker": ticker, "form": form,
                                                "status": "failed", "error": f"company: {e}",
                                                "last_attempt": _now()}
                stats["failed_company"] += 1
                logger.error(f"[{ticker}] Company lookup failed: {e}")
                continue

            process_ticker_form(company, ticker, form, years, manifest,
                                 revalidate=args.revalidate, stats=stats)

            if idx % 10 == 0:
                save_manifest(manifest)
                logger.info(f"[{form}] {idx}/{len(tickers)} tickers | "
                            f"+{stats['new_files']} new, {stats['skipped_complete']} skipped, "
                            f"{stats['failed_files'] + stats['failed_list'] + stats['failed_company']} failed")
            time.sleep(SEC_REQUEST_DELAY)

    save_manifest(manifest)
    logger.success(f"Done. new_files={stats['new_files']}, "
                   f"skipped_complete={stats['skipped_complete']}, "
                   f"failed_files={stats['failed_files']}, "
                   f"failed_list={stats['failed_list']}, failed_company={stats['failed_company']}")
    coverage_report(args.forms)


if __name__ == "__main__":
    main()
