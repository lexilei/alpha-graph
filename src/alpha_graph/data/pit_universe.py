"""Point-in-time S&P 500 membership for the evaluation panel.

Forward-bias fix ONLY: masking can remove current constituents' pre-join rows,
but the ~40% of 2011-2026 index members that later departed have no price or
filing data and cannot be restored (survivor bias remains; see task #9 and the
per-year gap in `adjudication_report`). Never describe a PIT delta as a
"correction" — print it with its avg cross-section and the departed-name gap.

Source: data/cache/sp500_pit_membership.csv (fja05680 reconstruction),
snapshots 1996->2026-01-14; a change-log after ~2018 (long gaps = no change).
Validation is monotonic dates + a cross-section band — NOT a gap rule (21
legitimate >35d gaps exist in-window; see plan v2 review E2).

Symbol drift: the CSV stores symbol-as-of-date for post-~2018 renames (FB, not
META) while older renames arrive back-normalized (WLP->ANTM, AA->ARNC already
applied upstream). RENAME_MAP carries the curated, verified map from old CSV
symbols to current panel tickers; entries are (panel_ticker, valid_until) —
`valid_until` bounds dated rules for symbol reuse (IR meant Ingersoll-Rand,
today's TT, through Feb 2020; the symbol was then reused by Gardner Denver,
today's panel IR). The map is a versioned research artifact: every change gets
a ledger line.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR

MEMBERSHIP_CSV = CACHE_DIR / "sp500_pit_membership.csv"
START = "2010-06-01"          # panel era with margin
CARRY_FORWARD_DAYS = 60       # max as-of carry past the last snapshot
XS_BAND = (490, 515)          # plausible S&P 500 cross-section, verified clean

# old CSV symbol -> (current panel ticker, valid_until or None)
# Verified against snapshot spans 2026-07-10 (see ledger). Chains are flattened
# (SYMC and NLOK both map to GEN). Replacements/acquisitions get NO entry.
RENAME_MAP: dict[str, tuple[str, str | None]] = {
    "FB":    ("META", None),   # 2022-06
    "ANTM":  ("ELV", None),    # 2022-06 (WLP already back-normalized upstream)
    "BLL":   ("BALL", None),   # 2022-05
    "RE":    ("EG", None),     # 2023-07
    "PKI":   ("RVTY", None),   # 2023-05
    "ABC":   ("COR", None),    # 2023-08
    "SYMC":  ("GEN", None),    # 2019-11 Symantec -> NortonLifeLock
    "NLOK":  ("GEN", None),    # 2022-11 NortonLifeLock -> Gen Digital
    "FLT":   ("CPAY", None),   # 2024-03
    "HCP":   ("DOC", None),    # 2019-11 HCP -> Healthpeak
    "PEAK":  ("DOC", None),    # 2024-03 Healthpeak -> DOC
    "COG":   ("CTRA", None),   # 2021-10
    "WLTW":  ("WTW", None),    # 2022-01
    "MYL":   ("VTRS", None),   # 2020-11
    "BHGE":  ("BKR", None),    # 2019-10
    "MWV":   ("SW", None),     # 2015-07 MeadWestvaco -> WestRock line
    "WRK":   ("SW", None),     # 2024-07 WestRock -> Smurfit Westrock
    "UTX":   ("RTX", None),    # 2020-04
    "HRS":   ("LHX", None),    # 2019-06
    "JEC":   ("J", None),      # 2019-12
    "TMK":   ("GL", None),     # 2019-08
    "MMC":   ("MRSH", None),   # 2026-01
    "CBS":   ("PSKY", None),   # 2019-12 CBS -> ViacomCBS listing line
    "VIAC":  ("PSKY", None),   # 2022-02 ViacomCBS -> Paramount
    "PARA":  ("PSKY", None),   # 2025-08 Paramount -> Paramount Skydance
    "ARNC":  ("HWM", None),    # 2020-04 (old-Alcoa line back-normalized to ARNC)
    "DISCA": ("WBD", None),    # 2022-04
    "DISCK": ("WBD", None),
    "FISV":  ("FI", None),     # 2023-06 (CSV double-lists both; set dedups)
    "BBT":   ("TFC", None),    # 2019-12 BB&T -> Truist (surviving listing)
    # Dated rule (symbol reuse): IR = Ingersoll-Rand plc (today's TT) through
    # Feb 2020; afterwards IR = the ex-Gardner-Denver company (panel IR).
    "IR":    ("TT", "2020-02-29"),
}


def _normalize(sym: str) -> str:
    return sym.strip().replace(".", "-").upper()


def load_membership(csv_path=MEMBERSHIP_CSV, start: str = START,
                    rename_map: dict | None = None) -> list[tuple[pd.Timestamp, frozenset]]:
    """Snapshots as [(date, frozenset of PANEL-space tickers)], renames applied.

    Validation: strictly increasing dates; cross-section within XS_BAND.
    """
    rename_map = RENAME_MAP if rename_map is None else rename_map
    m = pd.read_csv(csv_path)
    m["date"] = pd.to_datetime(m["date"])
    m = m[m["date"] >= pd.Timestamp(start)].sort_values("date").reset_index(drop=True)
    if not m["date"].is_monotonic_increasing or m["date"].duplicated().any():
        raise ValueError("membership snapshots are not strictly increasing")

    snaps: list[tuple[pd.Timestamp, frozenset]] = []
    for r in m.itertuples():
        raw = [_normalize(t) for t in r.tickers.split(",")]
        mapped = set()
        for t in raw:
            rule = rename_map.get(t)
            if rule is not None:
                new, until = rule
                if until is None or r.date <= pd.Timestamp(until):
                    mapped.add(new)
                else:
                    mapped.add(t)
            else:
                mapped.add(t)
        n = len(mapped)
        if not (XS_BAND[0] <= n <= XS_BAND[1]):
            raise ValueError(f"snapshot {r.date.date()}: cross-section {n} "
                             f"outside plausible band {XS_BAND}")
        snaps.append((r.date, frozenset(mapped)))
    logger.info(f"PIT membership: {len(snaps)} snapshots "
                f"{snaps[0][0].date()} -> {snaps[-1][0].date()}, renames applied")
    return snaps


def membership_asof(snaps: list[tuple[pd.Timestamp, frozenset]],
                    date: pd.Timestamp) -> frozenset | None:
    """Constituent set as of `date` (last snapshot <= date), or None when the
    date precedes the first snapshot or exceeds the carry-forward bound."""
    if date < snaps[0][0]:
        return None
    lo, hi = 0, len(snaps) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if snaps[mid][0] <= date:
            lo = mid
        else:
            hi = mid - 1
    d, s = snaps[lo]
    if lo == len(snaps) - 1 and (date - d).days > CARRY_FORWARD_DAYS:
        return None
    return s


def membership_mask(panel: pd.DataFrame,
                    snaps: list[tuple[pd.Timestamp, frozenset]] | None = None) -> pd.Series:
    """Boolean mask: row's ticker was an index member as of the row's date.

    Rows beyond the carry-forward bound are False (dropped from PIT mode)
    with a single warning.
    """
    if snaps is None:
        snaps = load_membership()
    mask = pd.Series(False, index=panel.index)
    n_beyond = 0
    for d, g in panel.groupby("date"):
        s = membership_asof(snaps, pd.Timestamp(d))
        if s is None:
            n_beyond += len(g)
            continue
        mask.loc[g.index] = g["ticker"].isin(s).values
    if n_beyond:
        logger.warning(f"PIT: {n_beyond} rows beyond membership coverage "
                       f"(before first snapshot or > {CARRY_FORWARD_DAYS}d "
                       f"past the last one) — excluded")
    return mask


def adjudication_report(panel: pd.DataFrame,
                        snaps: list[tuple[pd.Timestamp, frozenset]] | None = None) -> pd.DataFrame:
    """Per-year: panel names, PIT-member names among them, contemporaneous
    index size, and the departed-name gap (index members absent from the
    panel — the unrecoverable survivor hole). Also logs panel tickers that
    never match any snapshot (post-CSV joiners)."""
    if snaps is None:
        snaps = load_membership()
    panel_tk = set(panel["ticker"].unique())
    matched = set()
    for _, s in snaps:
        matched |= (panel_tk & s)
    never = sorted(panel_tk - matched)
    if never:
        logger.warning(f"PIT: {len(never)} panel tickers never match any "
                       f"snapshot (post-CSV joiners?): {never}")
    rows = []
    for year in sorted(panel["date"].dt.year.unique()):
        mid = pd.Timestamp(f"{year}-06-30")
        s = membership_asof(snaps, mid)
        if s is None:
            continue
        in_panel_year = set(panel.loc[panel["date"].dt.year == year, "ticker"])
        rows.append({
            "year": year,
            "index_size": len(s),
            "panel_members": len(in_panel_year & s),
            "departed_gap": len(s - panel_tk),
            "survivor_coverage": round(len(s & panel_tk) / len(s), 3),
        })
    return pd.DataFrame(rows)
