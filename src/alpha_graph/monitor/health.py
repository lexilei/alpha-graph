"""Inventory of data/cache/*.parquet — what exists, how big, how fresh.

Reads parquet FOOTER metadata only (row counts, schema, row-group column
statistics), never the row data, so the whole cache — including the ~107MB
market_data.parquet — inventories in well under a second. `deep=True` opts
into reading single columns for distinct counts.

The freshness flag is stated as the plain fact it is — "written before
market_data.parquet" — and not as a verdict. A signal cache older than the
price panel MAY be pre-repair (the 2026-07-14 price repair is the case that
motivates this) or may simply not depend on prices; the module reports the
ordering and leaves the reading to the human.

Usage:
    from alpha_graph.monitor.health import scan_cache
    for c in scan_cache():
        print(c.name, c.rows, c.date_range)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger

from alpha_graph.config import CACHE_DIR

# The panel every price-derived signal is built against; the freshness
# reference point.
REFERENCE_CACHE = "market_data.parquet"

# Column names treated as a row's event/observation date, most-specific first.
# `avail_date` outranks `filing_date`, and `filed` outranks `end`, so a cache
# carrying both a period and an availability stamp spans the one the panel
# actually merges on. Caches with none of these (summary tables, sector_map,
# permutation nulls) simply report no span — that is not an error.
_DATE_COLS = ("date", "avail_date", "filing_date", "acceptance_date",
              "event_date", "transaction_date", "entry_date", "seam_date",
              "veto_from", "month", "filed", "ddate", "period_end", "end")


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    rows: int
    columns: list[str]
    size_bytes: int
    modified: datetime
    date_col: str | None = None
    date_min: datetime | None = None
    date_max: datetime | None = None
    n_tickers: int | None = None      # deep scans only
    error: str | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_048_576

    @property
    def date_range(self) -> str | None:
        if self.date_min is None or self.date_max is None:
            return None
        return f"{self.date_min:%Y-%m-%d} → {self.date_max:%Y-%m-%d}"


@dataclass(frozen=True)
class CacheReport:
    entries: list[CacheEntry]
    cache_dir: Path
    reference: CacheEntry | None
    part_dirs: list[str]

    @property
    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self.entries)

    def older_than_reference(self) -> list[CacheEntry]:
        """Entries written before the reference price panel. A fact about
        write order, not a claim that they are wrong."""
        if self.reference is None:
            return []
        return [e for e in self.entries
                if e.path != self.reference.path
                and e.modified < self.reference.modified]

    def failed(self) -> list[CacheEntry]:
        return [e for e in self.entries if e.error]


def _pick_date_col(names: list[str]) -> str | None:
    for c in _DATE_COLS:
        if c in names:
            return c
    return None


def _column_span(pf: pq.ParquetFile, col: str) -> tuple[datetime | None, datetime | None]:
    """min/max of `col` from row-group statistics — no row data read.

    Returns (None, None) when the writer stored no statistics for the column,
    which is a normal parquet outcome, not an error.
    """
    md = pf.metadata
    try:
        idx = pf.schema_arrow.names.index(col)
    except ValueError:
        return None, None

    lo = hi = None
    for rg in range(md.num_row_groups):
        stats = md.row_group(rg).column(idx).statistics
        if stats is None or not stats.has_min_max:
            continue
        lo = stats.min if lo is None else min(lo, stats.min)
        hi = stats.max if hi is None else max(hi, stats.max)

    def _as_dt(v):
        if v is None or isinstance(v, datetime):
            return v
        try:  # date objects, and int64 epochs for some writers
            return datetime.fromisoformat(str(v))
        except (TypeError, ValueError):
            return None

    return _as_dt(lo), _as_dt(hi)


def inspect_parquet(path: Path, deep: bool = False) -> CacheEntry:
    stat = path.stat()
    base = dict(path=path, size_bytes=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime))
    try:
        pf = pq.ParquetFile(path)
        names = list(pf.schema_arrow.names)
        date_col = _pick_date_col(names)
        lo, hi = _column_span(pf, date_col) if date_col else (None, None)
        n_tickers = None
        if deep and "ticker" in names:
            n_tickers = len(pf.read(columns=["ticker"]).column("ticker").unique())
        return CacheEntry(rows=pf.metadata.num_rows, columns=names,
                          date_col=date_col, date_min=lo, date_max=hi,
                          n_tickers=n_tickers, **base)
    except Exception as exc:  # unreadable/corrupt file is itself the finding
        logger.warning(f"{path.name}: {exc}")
        return CacheEntry(rows=0, columns=[], error=str(exc), **base)


def scan_cache(cache_dir: Path | None = None, deep: bool = False) -> CacheReport:
    """Inventory every top-level parquet in `cache_dir`.

    Part directories (the per-ticker `*_parts/` shards the text builders write)
    are counted but not walked — they hold hundreds of files each and their
    merged parquet is already in the listing.
    """
    cache_dir = cache_dir or CACHE_DIR
    if not cache_dir.exists():
        return CacheReport(entries=[], cache_dir=cache_dir, reference=None, part_dirs=[])

    entries = [inspect_parquet(p, deep=deep)
               for p in sorted(cache_dir.glob("*.parquet"))]
    part_dirs = sorted(d.name for d in cache_dir.iterdir()
                       if d.is_dir() and not d.name.startswith("_"))
    reference = next((e for e in entries if e.name == REFERENCE_CACHE), None)
    return CacheReport(entries=entries, cache_dir=cache_dir,
                       reference=reference, part_dirs=part_dirs)
