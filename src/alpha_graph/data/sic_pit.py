"""Point-in-time SIC industry classification from EDGAR 10-K headers.

Cache: data/cache/sic_history.parquet, built by scripts/fetch_sic_history.py.
One row per corpus 10-K: the Standard Industrial Classification the company
carried in the EDGAR header of that filing — i.e. the classification as
publicly known ON the filing date. This is the PIT alternative to
sector_map.parquet, which is a current-day GICS-style snapshot (mildly
non-PIT for sector controls).

PIT rule (:func:`sic_asof`): the SIC on date D is the code of the latest 10-K
with ``filing_date <= D`` (boundary inclusive); None before the first 10-K.
Resolution is annual — a mid-year reclassification only becomes visible at
the next 10-K, which is exactly what was knowable from headers at the time.

Granularity: :func:`sic_division` maps a 4-digit code onto the standard SIC
divisions (A-J range table below) — ~10 buckets, comparable to the 11 GICS
sectors, the natural drop-in for sector-neutralization controls.
:func:`sic2` returns the 2-digit major group (~60 buckets, between GICS
sector and industry). Unknown/missing/out-of-range codes map to "UNKNOWN"
(EDGAR uses 0000 for unassigned; 1800-1999, 6800-6999, 9730-9899 are gaps
in the official table).

NOT wired into the panel or the judge: the sector-neutral evaluation
convention is frozen on the GICS snapshot; switching sector definitions is
a registered convention change handled separately.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_graph.config import CACHE_DIR

SIC_HISTORY_PATH = CACHE_DIR / "sic_history.parquet"

_HIST_COLS = ("filing_date", "sic_code")

# Official SIC division ranges (inclusive), 4-digit code space.
_DIVISIONS: tuple[tuple[int, int, str], ...] = (
    (100, 999, "Agriculture"),           # A: agriculture, forestry, fishing
    (1000, 1499, "Mining"),              # B
    (1500, 1799, "Construction"),        # C
    (2000, 3999, "Manufacturing"),       # D
    (4000, 4999, "Transport & Utilities"),  # E: transport, communications, utilities
    (5000, 5199, "Wholesale"),           # F
    (5200, 5999, "Retail"),              # G
    (6000, 6799, "Finance"),             # H: finance, insurance, real estate
    (7000, 8999, "Services"),            # I
    (9100, 9729, "Public Admin"),        # J
    (9900, 9999, "Nonclassifiable"),     # major group 99 (EDGAR: 9995)
)

UNKNOWN = "UNKNOWN"


def load_sic_history(path: Path | None = None) -> pd.DataFrame:
    """Load the SIC history cache, usable rows only (sic_code present).

    Rows with NA sic_code exist in the parquet as fetch bookkeeping (header
    had no SIC line) and are dropped here.
    """
    return _load_cached(str(path or SIC_HISTORY_PATH)).copy()


@lru_cache(maxsize=2)
def _load_cached(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["sic_code"]).copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df.sort_values(["ticker", "filing_date"], kind="mergesort").reset_index(drop=True)


def _code_int(code) -> int | None:
    """4-digit SIC as int, or None for anything missing/invalid/unassigned."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    if isinstance(code, (int, np.integer)):
        n = int(code)
    elif isinstance(code, float):
        n = int(code)
    else:
        s = str(code).strip()
        if not s.isdigit():
            return None
        n = int(s)
    return n if 0 < n <= 9999 else None  # 0000 = EDGAR "unassigned"


def sic_division(code) -> str:
    """Standard SIC division for a 4-digit code; "UNKNOWN" if unmappable."""
    n = _code_int(code)
    if n is None:
        return UNKNOWN
    for lo, hi, name in _DIVISIONS:
        if lo <= n <= hi:
            return name
    return UNKNOWN


def sic2(code) -> str:
    """2-digit SIC major group (zero-padded str); "UNKNOWN" if unmappable."""
    n = _code_int(code)
    return UNKNOWN if n is None else f"{n:04d}"[:2]


def _history_for(ticker_or_frame: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(ticker_or_frame, str):
        df = _load_cached(str(SIC_HISTORY_PATH))
        return df[df["ticker"] == ticker_or_frame]
    frame = ticker_or_frame
    missing = [c for c in _HIST_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"history frame lacks columns {missing}")
    if "ticker" in frame.columns and frame["ticker"].nunique() > 1:
        raise ValueError(
            "history frame holds multiple tickers; pass a ticker string or a "
            "single-ticker frame"
        )
    return frame


def sic_asof(
    ticker_or_frame: str | pd.DataFrame,
    dates,
) -> str | None | pd.Series:
    """SIC code from the latest 10-K filed on or before each date.

    Parameters
    ----------
    ticker_or_frame : ticker string (looked up in the cache) or a history
        DataFrame with columns filing_date/sic_code for a single entity.
    dates : scalar date-like, or a list-like of dates.

    Returns a str (or None if no 10-K yet) for a scalar date, else an
    object-dtype pd.Series indexed by ``dates``. Availability is by filing
    date, boundary INCLUSIVE: a 10-K filed on D counts on D. Same-day
    duplicates resolve to the highest accession (deterministic).
    """
    hist = _history_for(ticker_or_frame)
    hist = hist.dropna(subset=["sic_code"])

    scalar = np.ndim(dates) == 0 and not isinstance(dates, (list, tuple, set))
    dt = pd.DatetimeIndex(pd.to_datetime([dates] if scalar else list(dates)))

    if hist.empty:
        out: list[str | None] = [None] * len(dt)
    else:
        sort_cols = ["filing_date"] + (["accession"] if "accession" in hist.columns else [])
        h = hist.sort_values(sort_cols, kind="mergesort")
        filed = h["filing_date"].values
        codes = h["sic_code"].to_numpy(dtype=object)
        idx = np.searchsorted(filed, dt.values, side="right") - 1
        out = [codes[i] if i >= 0 else None for i in idx]
    if scalar:
        return out[0]
    return pd.Series(out, index=dt, name="sic_code", dtype=object)
