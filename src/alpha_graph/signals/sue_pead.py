"""C17 `sue_pead`: standardized unexpected earnings (SUE) for PEAD.

SUE = (EPS_q − EPS_{q−4}) / std(trailing 8 seasonal diffs, min 6), built from
AS-FILED diluted EPS in `xbrl_facts.parquet` (SEC Financial Statement Data
Sets: each value frozen as originally filed). Registered 2026-07-13
(reports/factor_preregistration.md; FACTORS.md C17). Construction, exactly
as registered:

- EPS tag `EarningsPerShareDiluted` ONLY — no silent basic fallback; per-share
  uom rows (FSDS stores per-share values with uom "USD"; the handful of other
  uoms are filer tagging errors, dropped and logged).
- First-filed discipline: for each (cik, ddate, qtrs) the value from the
  EARLIEST `filed` accession (ties broken by accepted, then adsh) — the
  as-registered number, never a post-hoc revision. Splits therefore appear
  as level breaks between fiscal years (inherent to the registered design).
  Quarterly EPS = qtrs=1 rows.
- Q4 derivation where no standalone Q4 exists: annual EPS (qtrs=4, ddate =
  fiscal-year end = the 10-K's own period) minus the sum of that fiscal
  year's three qtrs=1 values (all first-filed; the three quarter-ends must
  sit 45-330 days before the FY end). Skipped when any of the three is
  missing. Flagged `derived_q4`; availability = the ANNUAL filing's
  availability (the 10-K discloses the derivation).
- Seasonal diff: quarters ordered per cik by ddate; q−4 = the quarter with
  ddate 365±45 days earlier (nearest to 365 when several qualify).
- SUE = diff / std(TRAILING 8 diffs EXCLUDING the current one, min 6,
  ddof=1); std == 0 -> NaN. The window is positional over the firm's
  observed quarters. (The registration says "std(last 8 seasonal diffs)"
  without stating inclusion; pinned 2026-07-13 pre-evaluation as EXCLUDING
  the current diff — non-overlapping null, consistent with C15's design;
  ledgered as a build-time pin.)
- Availability (`disclosure_date`) = the LAST Item-2.02 8-K date in
  (period_end, statement filed]. Corrected 2026-07-14 (data review; the
  registration said "earliest"): guidance/preliminary 8-Ks legitimately
  carry Item 2.02 without the final EPS — the first-2.02 rule dated 132
  SUE rows a median 23 days before their EPS was computable (e.g. AAPL's
  2018-12-31 quarter got the 2019-01-03 guidance-cut letter instead of the
  2019-01-29 earnings release). The last 2.02 in the window errs stale,
  never early. Item labels are not in the filing JSONs' section keys (all
  `full_text`); they are recovered from the text's "Item 2.02" headers —
  the same extraction the C12-C14 event factors used. If that 8-K was
  ACCEPTED at/after 16:00 ET, the availability moves to the next calendar
  day (it wasn't public before that close). No qualifying 8-K -> the
  statement's filed date, unshifted (a filed date is already the
  availability day for EDGAR docs used at close). The v0 t+1 availability
  lag is applied at the panel merge on top of this, like every other
  filing-dated factor.

Cache: data/cache/sue_pead.parquet — ticker, disclosure_date, period_end,
sue, eps_q, eps_q4_ago, derived_q4, via_8k02. A row exists for every scored
quarter (sue is NaN until the min-6 history gate passes); the panel merge
drops NaN-sue rows.

Usage:
    python -m alpha_graph.signals.sue_pead
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR

EPS_TAG = "EarningsPerShareDiluted"
PER_SHARE_UOM = "USD"        # FSDS unit for per-share values
SEASONAL_DAYS = 365          # target gap to the same fiscal quarter last year
SEASONAL_TOL_DAYS = 45       # match tolerance around SEASONAL_DAYS
STD_WINDOW = 8               # trailing seasonal diffs in the SUE denominator
STD_MIN_OBS = 6              # minimum diffs before SUE is defined
FY_QTR_MIN_DAYS = 45         # window (days before FY end) holding Q1-Q3 ends
FY_QTR_MAX_DAYS = 330        # (prior FY end at ~364-371d stays outside)
ACCEPT_CUTOFF_HOUR = 16      # ET; accepted at/after this -> next-day availability
ITEM_202_RE = re.compile(r"item\s+2\.02", re.IGNORECASE)

CACHE_NAME = "sue_pead.parquet"


# --------------------------------------------------------------------------- #
# EPS extraction
# --------------------------------------------------------------------------- #

def first_filed(facts: pd.DataFrame, tag: str,
                uom: str = PER_SHARE_UOM) -> pd.DataFrame:
    """Facts of one tag/uom, one row per (cik, ddate, qtrs): the first-filed
    value.

    Sort by (filed, accepted, adsh) and keep the first row per key, so a
    later accession restating the same period (amendment or next year's
    comparative) never replaces the original. The as-filed discipline shared
    by C17 (diluted EPS) and the B8/B9 fundamentals controls
    (signals/fundamentals_pit.py: equity, net income).
    """
    sel = facts[facts["tag"] == tag]
    keep = sel[sel["uom"] == uom].copy()
    if len(sel) - len(keep):
        logger.info(f"dropped {len(sel) - len(keep)} {tag} rows with "
                    f"uom != {uom} (filer tagging errors)")
    keep = keep.sort_values(["filed", "accepted", "adsh"], kind="mergesort")
    return keep.drop_duplicates(subset=["cik", "ddate", "qtrs"], keep="first")


def first_filed_eps(facts: pd.DataFrame) -> pd.DataFrame:
    """Diluted-EPS facts, one row per (cik, ddate, qtrs): the first-filed
    value (FSDS stores per-share values with uom "USD"; other uoms are filer
    tagging errors, dropped and logged)."""
    return first_filed(facts, EPS_TAG, PER_SHARE_UOM)


def quarterly_eps(first: pd.DataFrame) -> pd.DataFrame:
    """Standalone quarterly EPS rows plus derived-Q4 rows.

    Returns one row per (cik, quarter): cik, ticker, ddate, eps, derived_q4,
    stmt_adsh, stmt_filed (the statement that made the value computable —
    for derived Q4 that is the ANNUAL filing).

    Value-agnostic despite the `eps` column name: fundamentals_pit.py feeds
    first-filed NetIncomeLoss through it for the B9 TTM control.
    """
    q = first.loc[first["qtrs"] == 1,
                  ["cik", "ticker", "ddate", "value", "adsh", "filed"]].rename(
        columns={"value": "eps", "adsh": "stmt_adsh", "filed": "stmt_filed"})
    q["derived_q4"] = False

    # Own-fiscal-year annuals: qtrs=4 rows whose ddate is the 10-K's own period
    # (comparative prior-FY rows in later filings have ddate < period and lose
    # the first-filed race anyway).
    ann = first[(first["qtrs"] == 4) & (first["ddate"] == first["period"])
                & first["form"].str.startswith("10-K")]

    have_q = set(zip(q["cik"], q["ddate"]))
    by_cik = {c: g.sort_values("ddate") for c, g in q.groupby("cik", sort=False)}
    derived, n_skipped = [], 0
    for r in ann.itertuples():
        if (r.cik, r.ddate) in have_q:
            continue                                   # standalone Q4 exists
        g = by_cik.get(r.cik)
        if g is None:
            n_skipped += 1
            continue
        off = (r.ddate - g["ddate"]).dt.days
        fy = g[(off >= FY_QTR_MIN_DAYS) & (off <= FY_QTR_MAX_DAYS)]
        if len(fy) != 3:                               # a quarter missing/extra
            n_skipped += 1
            continue
        derived.append({
            "cik": r.cik, "ticker": r.ticker, "ddate": r.ddate,
            "eps": r.value - fy["eps"].sum(),
            "stmt_adsh": r.adsh, "stmt_filed": r.filed, "derived_q4": True,
        })
    logger.info(f"Q4 derivation: {len(derived)} derived, {n_skipped} skipped "
                f"(missing quarters), {len(ann)} own-FY annuals")
    if derived:
        q = pd.concat([q, pd.DataFrame(derived)], ignore_index=True)
    return q.sort_values(["cik", "ddate"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Seasonal diff + SUE
# --------------------------------------------------------------------------- #

def add_sue(qeps: pd.DataFrame) -> pd.DataFrame:
    """Add eps_q4_ago, and sue = seasonal diff / std(trailing 8 diffs excl.
    current, min 6, ddof=1). Quarters ordered per cik by ddate."""
    lo_gap = np.timedelta64(SEASONAL_DAYS - SEASONAL_TOL_DAYS, "D")
    hi_gap = np.timedelta64(SEASONAL_DAYS + SEASONAL_TOL_DAYS, "D")
    target = np.timedelta64(SEASONAL_DAYS, "D")

    parts = []
    for _, g in qeps.groupby("cik", sort=False):
        g = g.sort_values("ddate").reset_index(drop=True)
        dd = g["ddate"].to_numpy()
        eps = g["eps"].to_numpy()
        prev = np.full(len(g), np.nan)
        lo = np.searchsorted(dd, dd - hi_gap, side="left")
        hi = np.searchsorted(dd, dd - lo_gap, side="right")
        for i in range(len(g)):
            if hi[i] <= lo[i]:
                continue                                # no match in tolerance
            cand = np.arange(lo[i], hi[i])
            j = cand[np.argmin(np.abs((dd[i] - dd[cand]) - target))]
            prev[i] = eps[j]
        g["eps_q4_ago"] = prev
        diff = g["eps"] - g["eps_q4_ago"]
        std8 = diff.shift(1).rolling(STD_WINDOW, min_periods=STD_MIN_OBS).std(ddof=1)
        g["sue"] = diff / std8.where(std8 != 0)         # std==0 -> NaN
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Availability: Item-2.02 8-K, else statement filed date
# --------------------------------------------------------------------------- #

def scan_8k_202() -> pd.DataFrame:
    """One row per 8-K filing in the corpus: ticker, fdate, accession, has_202.

    Item labels come from the filing text ("Item 2.02" headers; the JSONs'
    section keys are all `full_text`, so keys alone carry no item labels).
    """
    rows = []
    for d in sorted(FILINGS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        for f in d.glob("8-K_*.json"):
            parts = f.name.split("_")
            if len(parts) < 3:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.debug(f"[{d.name}] failed to parse {f.name}")
                continue
            text = "\n\n".join(data.get("sections", {}).values())
            rows.append((d.name, parts[1], parts[2].removesuffix(".json"),
                         bool(ITEM_202_RE.search(text))))
    df = pd.DataFrame(rows, columns=["ticker", "fdate", "accession", "has_202"])
    df["fdate"] = pd.to_datetime(df["fdate"], errors="coerce")
    df = df.dropna(subset=["fdate"])
    logger.info(f"8-K scan: {len(df)} filings, {int(df['has_202'].sum())} "
                f"with Item 2.02 ({df['ticker'].nunique()} tickers)")
    return df


def add_availability(qeps: pd.DataFrame, eightk: pd.DataFrame,
                     acceptance: pd.DataFrame) -> pd.DataFrame:
    """Add disclosure_date and via_8k02.

    disclosure_date = the LAST Item-2.02 8-K date in (period_end,
    stmt_filed], pushed one calendar day when that 8-K was accepted at/after
    16:00 ET (join on digits-only accession); else stmt_filed (no shift).

    Last, not first (corrected 2026-07-14): guidance/preliminary 8-Ks carry
    Item 2.02 without the final EPS, so the first 2.02 in the window can
    predate the number's existence; the last one errs stale, never early.
    Same-date ties resolve to the latest acceptance_ts (NaT last). The
    statement-filed fallback stays the terminal backstop.
    """
    e = eightk.loc[eightk["has_202"].astype(bool)].merge(
        acceptance[["accession", "acceptance_ts"]].drop_duplicates("accession"),
        on="accession", how="left")
    e = e.sort_values(["fdate", "acceptance_ts"], kind="mergesort")
    by_t = {t: (g["fdate"].to_numpy(), g["acceptance_ts"].to_numpy())
            for t, g in e.groupby("ticker", sort=False)}
    n_missing_ts = int(e["acceptance_ts"].isna().sum())
    if n_missing_ts:
        logger.warning(f"{n_missing_ts} Item-2.02 8-Ks lack an acceptance "
                       f"timestamp — used without the 16:00 shift")

    out = qeps.copy()
    disc = out["stmt_filed"].to_numpy().astype("datetime64[ns]").copy()
    via = np.zeros(len(out), dtype=bool)
    tickers = out["ticker"].to_numpy()
    pends = out["ddate"].to_numpy().astype("datetime64[ns]")
    fileds = out["stmt_filed"].to_numpy().astype("datetime64[ns]")
    day = np.timedelta64(1, "D")
    for i in range(len(out)):
        got = by_t.get(tickers[i])
        if got is None:
            continue
        dates, ts = got
        lo = np.searchsorted(dates, pends[i], side="right")   # first > period_end
        hi = np.searchsorted(dates, fileds[i], side="right")  # first > stmt_filed
        if hi > lo:
            k = hi - 1                         # LAST 2.02 in (period_end, filed]
            d = dates[k]
            t = pd.Timestamp(ts[k])
            if not pd.isna(t) and t.hour >= ACCEPT_CUTOFF_HOUR:
                d = d + day                    # not public before that close
            disc[i] = d
            via[i] = True
    out["disclosure_date"] = disc
    out["via_8k02"] = via
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def compute(
    facts: pd.DataFrame | None = None,
    eightk: pd.DataFrame | None = None,
    acceptance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the C17 cache. All three inputs are injectable for tests;
    defaults read the real artifacts, and only real runs write the cache."""
    inject = not (facts is None and eightk is None and acceptance is None)
    if facts is None:
        facts = pd.read_parquet(CACHE_DIR / "xbrl_facts.parquet")
    if eightk is None:
        eightk = scan_8k_202()
    if acceptance is None:
        acceptance = pd.read_parquet(CACHE_DIR / "filing_acceptance.parquet")

    first = first_filed_eps(facts)
    n_all = facts["ticker"].nunique()
    nq = first[first["qtrs"] == 1]
    logger.info(f"diluted-EPS coverage: {nq['ticker'].nunique()}/{n_all} corpus "
                f"tickers, {len(nq)} first-filed quarterly observations")

    qeps = quarterly_eps(first)
    qeps = add_sue(qeps)
    qeps = add_availability(qeps, eightk, acceptance)

    df = qeps.rename(columns={"ddate": "period_end"})[
        ["ticker", "disclosure_date", "period_end", "sue", "eps",
         "eps_q4_ago", "derived_q4", "via_8k02"]
    ].rename(columns={"eps": "eps_q"})
    df = df.sort_values(["ticker", "disclosure_date", "period_end"],
                        kind="mergesort").reset_index(drop=True)

    logger.info(
        f"sue_pead: {len(df)} quarter rows, {df['ticker'].nunique()} tickers; "
        f"{df['derived_q4'].mean() * 100:.1f}% derived Q4; "
        f"{df['via_8k02'].mean() * 100:.1f}% via 8-K 2.02; "
        f"{df['sue'].notna().mean() * 100:.1f}% with defined SUE"
    )
    if not inject:
        path = CACHE_DIR / CACHE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} rows to {path}")
    return df


def main():
    p = argparse.ArgumentParser(description="Build the C17 sue_pead cache")
    p.parse_args()
    df = compute()
    if not df.empty:
        print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
