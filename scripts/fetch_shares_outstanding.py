"""Build data/cache/shares_outstanding_pit.parquet: point-in-time shares outstanding.

For every ticker under data/filings/ we pull the XBRL share-count history from
the SEC company-facts APIs and date each count by the FILING date that disclosed
it (`filed`), not the measurement date (`end`). That makes the table point-in-
time: a share count is usable on date D only if filed <= D. The loader with
those semantics lives in alpha_graph.data.shares_pit (shares_asof /
market_cap_panel).

Source concepts, in preference order (recorded per row in `concept`):
    dei:EntityCommonStockSharesOutstanding                 cover page count
    us-gaap:CommonStockSharesOutstanding                   balance-sheet parenthetical
    us-gaap:CommonStockSharesIssued                        includes treasury
    us-gaap:WeightedAverageNumberOfSharesOutstandingBasic  EPS denominator (proxy)
The small per-concept endpoint (companyconcept) is tried first; the full
companyfacts blob is fetched only when dei is missing (404) or thin. Thinness
matters because multi-class filers (Alphabet, Fox, Meta, News Corp, ...) tag
cover-page counts per share class WITH a class-member axis, and the facts APIs
drop dimensioned facts entirely — their dei history is empty or near-empty and
the combined count must come from an undimensioned us-gaap tag. Filers that
dimension the balance-sheet tags too but present one combined EPS (META, HRL,
LEN, ...) still expose the basic-EPS share count — a within-period average
rather than a point count, close enough for market-cap use and flagged by its
concept name. It is a DURATION concept: a 10-Q carries quarter and YTD
averages ending on the same date, so within each filing we keep the shortest
period (latest `start`) and never sum across periods.

SCALE MISTAGS: filers occasionally tag a count at the wrong scale (PKG has
lone x1000 rows: 8.99e10 where the neighbors say 8.99e7). Rows more than 100x
away from the ticker's median count (log10 distance > 2) are dropped at build
time. CAUTION — real splits are NOT bounded by that band: reverse splits up
to 1-for-200 exist in this universe (EXE, then Chesapeake, April 2020 — a
200x level move), and EXE's true post-split rows already sit 67x from its
median, only ~1.5x inside the drop threshold. A name with a longer
post-reverse-split history (median pulled further from the old regime) would
have TRUE rows falsely dropped. As a tripwire, any would-drop row on a
ticker whose raw series carries a real-split signature (>=50x persistent
level break; fires with as few as 3 rows beyond the break, 2 for breaks
>=100x — has_split_signature) is WARNING-logged for operator hand-checking
before being dropped anyway.

MULTI-CLASS RESOLUTION: when one filing reports several classes as separate
facts they surface as duplicate (end, filed) rows that must be SUMMED
(GOOGL A+B+C). But duplicate (end, filed) rows can also come from DIFFERENT
filings (same-day re-report/amendment), where summing would double-count. We
therefore sum within an accession (one filing's cover page) and, across
accessions sharing (end, filed), keep the latest accession only. Rows built
by summing >1 class fact carry class_summed=True so they can be audited.

Ticker -> CIK reuses the map already fetched into
data/cache/filing_acceptance.parquet (falls back to company_tickers.json for
tickers that table folded into a sibling share class, e.g. GOOGL -> GOOG's
CIK). Tickers with two CIKs (XOM's 2026 holdco reorg) get both histories,
deduplicated on (end, filed) with the longer history winning collisions.

Output columns:
    ticker        corpus ticker (str)
    cik           10-digit zero-padded CIK (str)
    end           measurement date (datetime)
    filed         SEC filing date = availability date (datetime)
    shares        shares outstanding (float, classes summed)
    concept       XBRL concept the row came from (str)
    form          form type of the disclosing filing (str)
    class_summed  True where >1 class fact was summed (bool)

Usage:
    python scripts/fetch_shares_outstanding.py             # skip if cache <7 days old
    python scripts/fetch_shares_outstanding.py --refresh   # force re-fetch
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR
from alpha_graph.data.shares_pit import SHARES_PIT_PATH, shares_asof

ACCEPTANCE_PATH = CACHE_DIR / "filing_acceptance.parquet"
MARKET_DATA_PATH = CACHE_DIR / "market_data.parquet"
CONCEPT_URL = ("https://data.sec.gov/api/xbrl/companyconcept/"
               "CIK{cik}/dei/EntityCommonStockSharesOutstanding.json")
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "Lei Zhihan leizhihan0606@gmail.com"  # SEC fair-access identity
CONCEPT_ORDER = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesIssued"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
]
# Another fetcher may hit data.sec.gov concurrently: self-limit to 4 req/s
# (SEC's fair-access ceiling is 10/s across ALL of a user's processes).
REQUEST_INTERVAL = 0.25
MAX_AGE_DAYS = 7.0
THIN_GROUPS = 8  # dei histories shorter than this trigger the companyfacts fallback

_last_request_t = 0.0


# --------------------------------------------------------------------------- #
# HTTP (rate-limited, retry once on 429/5xx, None on 404)
# --------------------------------------------------------------------------- #

def _get_json(session: requests.Session, url: str) -> dict | None:
    global _last_request_t
    for attempt in (0, 1):
        wait = REQUEST_INTERVAL - (time.monotonic() - _last_request_t)
        if wait > 0:
            time.sleep(wait)
        _last_request_t = time.monotonic()
        try:
            resp = session.get(url, timeout=60)
        except requests.RequestException as e:
            if attempt == 1:
                raise
            logger.warning(f"{url}: {e}; retrying in 2s")
            time.sleep(2)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == 1:
                resp.raise_for_status()
            backoff = float(resp.headers.get("Retry-After", 5))
            logger.warning(f"{url}: HTTP {resp.status_code}; retrying in {backoff:.0f}s")
            time.sleep(backoff)
            continue
        resp.raise_for_status()
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Ticker -> CIK(s)
# --------------------------------------------------------------------------- #

def corpus_tickers() -> list[str]:
    return sorted(d.name for d in FILINGS_DIR.iterdir()
                  if d.is_dir() and not d.name.startswith((".", "_")))


def resolve_ciks(session: requests.Session, tickers: list[str]) -> dict[str, list[str]]:
    """ticker -> list of 10-digit CIKs (usually one; XOM's reorg has two).

    Reuses the map fetch_acceptance_datetimes.py already built (it covers
    renames/reorgs via its accession-prefix pass); company_tickers.json fills
    tickers that table folded into a sibling class (GOOGL/FOXA/NWSA).
    """
    resolved: dict[str, list[str]] = {}
    if ACCEPTANCE_PATH.exists():
        pairs = (pd.read_parquet(ACCEPTANCE_PATH, columns=["ticker", "cik"])
                 .drop_duplicates())
        for t, grp in pairs.groupby("ticker"):
            resolved[t] = sorted(grp["cik"])
        logger.info(f"Reused {len(resolved)} ticker->CIK entries from {ACCEPTANCE_PATH.name}")

    misses = [t for t in tickers if t not in resolved]
    if misses:
        data = _get_json(session, COMPANY_TICKERS_URL) or {}
        sec_map = {row["ticker"].upper(): f"{int(row['cik_str']):010d}"
                   for row in data.values()}
        for t in list(misses):
            for cand in (t, t.replace(".", "-"), t.replace("-", ".")):
                if cand.upper() in sec_map:
                    resolved[t] = [sec_map[cand.upper()]]
                    misses.remove(t)
                    break
    if misses:
        logger.warning(f"{len(misses)} tickers failed CIK resolution: {misses}")
    return {t: resolved[t] for t in tickers if t in resolved}


# --------------------------------------------------------------------------- #
# Fact fetching
# --------------------------------------------------------------------------- #

def _parse_facts(entries: list[dict], concept: str) -> pd.DataFrame:
    rows = [
        {
            "end": e.get("end"),
            "val": e.get("val"),
            "filed": e.get("filed"),
            "form": e.get("form"),
            "frame": e.get("frame"),
            "accn": e.get("accn"),
            "start": e.get("start"),  # duration concepts only
            "concept": concept,
        }
        for e in entries
        if e.get("val") is not None and e.get("end") and e.get("filed")
    ]
    return pd.DataFrame(rows)


def _n_groups(df: pd.DataFrame) -> int:
    return 0 if df.empty else len(df[["end", "filed"]].drop_duplicates())


def fetch_cik_facts(session: requests.Session, cik: str) -> pd.DataFrame:
    """Share-count facts for one CIK from the best available concept.

    Tries the small dei companyconcept endpoint first; on 404 or a thin
    history falls back to full companyfacts and takes the first concept in
    CONCEPT_ORDER with a usable history (>= THIN_GROUPS distinct
    (end, filed) groups), else the longest one.
    """
    j = _get_json(session, CONCEPT_URL.format(cik=cik))
    dei = _parse_facts((j or {}).get("units", {}).get("shares", []),
                       "dei:EntityCommonStockSharesOutstanding")
    if _n_groups(dei) >= THIN_GROUPS:
        return dei

    jf = _get_json(session, FACTS_URL.format(cik=cik))
    facts = (jf or {}).get("facts", {})
    candidates = [dei]
    for taxonomy, tag in CONCEPT_ORDER[1:]:
        entries = (facts.get(taxonomy, {}).get(tag, {})
                   .get("units", {}).get("shares", []))
        candidates.append(_parse_facts(entries, f"{taxonomy}:{tag}"))
    for cand in candidates:  # preference order, first usable history wins
        if _n_groups(cand) >= THIN_GROUPS:
            return cand
    return max(candidates, key=_n_groups)


def resolve_multiclass(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (end, filed); see module docstring for the rule."""
    if df["start"].notna().any():
        # Duration concept (weighted-average shares): a filing reports
        # several periods ending the same date (quarter + YTD + prior year);
        # keep the shortest one per filing — closest to a point count.
        df = (df.sort_values("start", na_position="first")
              .drop_duplicates(subset=["end", "filed", "accn"], keep="last"))
    per_accn = (df.groupby(["end", "filed", "accn"], dropna=False)
                .agg(shares=("val", "sum"), n_class=("val", "size"),
                     form=("form", "first"), concept=("concept", "first"))
                .reset_index())
    # Across filings sharing (end, filed) keep the latest accession
    # (accession numbers of one filer sort chronologically within a day).
    resolved = (per_accn.sort_values("accn", kind="mergesort")
                .groupby(["end", "filed"], as_index=False).last())
    resolved["class_summed"] = resolved["n_class"] > 1
    return resolved.drop(columns=["accn", "n_class"])


def has_split_signature(shares, min_ratio: float = 50.0,
                        min_side: int = 4) -> bool:
    """True when a time-ordered share-count series contains a >=min_ratio
    persistent level break — the signature of a real (reverse) split, as
    opposed to a lone scale mistag.

    At each cut with min_side rows on both sides OF THE CUT, compare the
    MEDIANS of the min_side rows just before vs just after the cut. A lone
    mistag spike cannot pull a 4-row median to its scale, so PKG-style x1000
    rows do not trip; EXE's 1-for-200 reverse split does. Note the window
    may straddle the BREAK: a 4-row median is decided by its middle two
    values, so 3 rows beyond the break suffice, and a half-straddling
    window's midpoint median lets a >=2x-min_ratio break fire with just 2 —
    NOT the ">=4 rows on both sides of the break" this docstring once
    claimed. Safe direction: this gate only ADDS hand-check warnings, never
    drops. Windows are local because a
    post-split regime can be short (EXE re-issued shares in its 2021
    bankruptcy reorg after only 4 filings at the post-split count) —
    whole-side medians would dilute such a break away.
    """
    s = np.asarray(shares, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    for cut in range(min_side, len(s) - min_side + 1):
        left = np.median(s[cut - min_side:cut])
        right = np.median(s[cut:cut + min_side])
        if max(left, right) >= min_ratio * min(left, right):
            return True
    return False


def drop_scale_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop per-ticker scale mistags (see module docstring): rows whose count
    sits more than 100x from the ticker's median count, plus non-positive
    counts. Mistags are lone x1000 spikes; splits are persistent level shifts
    that usually stay inside 100x of the median — but not by construction:
    EXE's 1-for-200 reverse split leaves its true post-split rows 67x out,
    only ~1.5x inside the threshold. Would-drop rows on a ticker whose raw
    series shows a real-split signature (has_split_signature) are therefore
    WARNING-logged for hand-checking; they are still dropped."""
    ok = df["shares"] > 0
    med = df["shares"].where(ok).groupby(df["ticker"]).transform("median")
    dev = (np.log10(df["shares"].where(ok)) - np.log10(med)).abs()
    bad = ~ok | (dev > 2)
    if bad.any():
        per = df.loc[bad, "ticker"].value_counts().to_dict()
        logger.warning(f"dropped {int(bad.sum())} scale-outlier rows "
                       f"(>100x off ticker median): {per}")
        suspects = [
            t for t in sorted(per)
            if has_split_signature(
                df.loc[df["ticker"] == t].sort_values(["end", "filed"])["shares"])
        ]
        if suspects:
            logger.warning(
                f"would-drop rows on {suspects}, whose raw series shows a "
                f"real-split signature (>=50x level break, >=4 rows on both "
                f"sides) — the 100x-median rule may be removing TRUE "
                f"post-split rows there; hand-check these tickers. "
                f"Dropping anyway.")
    return df.loc[~bad]


def build_table(tickers: list[str]) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})

    cik_map = resolve_ciks(session, tickers)
    frames: list[pd.DataFrame] = []
    empty: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        per_cik: list[pd.DataFrame] = []
        for cik in cik_map.get(ticker, []):
            try:
                raw = fetch_cik_facts(session, cik)
            except Exception as e:  # noqa: BLE001
                logger.error(f"{ticker} CIK {cik}: fetch failed: {e}")
                continue
            if raw.empty:
                continue
            resolved = resolve_multiclass(raw)
            resolved.insert(0, "cik", cik)
            per_cik.append(resolved)
        if not per_cik:
            empty.append(ticker)
        else:
            # Two CIKs (reorgs): longer history wins (end, filed) collisions.
            merged = (pd.concat(sorted(per_cik, key=len, reverse=True))
                      .drop_duplicates(subset=["end", "filed"], keep="first"))
            merged.insert(0, "ticker", ticker)
            frames.append(merged)
        if i % 50 == 0:
            logger.info(f"{i}/{len(tickers)} tickers fetched")

    if empty:
        logger.warning(f"{len(empty)} tickers returned no share facts: {empty}")
    df = pd.concat(frames, ignore_index=True)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["shares"] = df["shares"].astype(float)
    df = drop_scale_outliers(df)
    df = df.sort_values(["ticker", "filed", "end"]).reset_index(drop=True)
    return df[["ticker", "cik", "end", "filed", "shares", "concept", "form",
               "class_summed"]]


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #

def _split_check(df: pd.DataFrame, ticker: str, effective: str,
                 ratio_lo: float, ratio_hi: float) -> None:
    """Report the share-count jump across a split and when it became known."""
    eff = pd.Timestamp(effective)
    sub = (df[df["ticker"] == ticker].groupby(["end", "filed"], as_index=False)
           ["shares"].sum().sort_values("end"))
    pre = sub[sub["end"] < eff].tail(1)
    post = sub[sub["end"] >= eff].head(1)
    if pre.empty or post.empty:
        print(f"  {ticker} split {effective}: MISSING pre/post rows -> FAIL")
        return
    ratio = float(post["shares"].iloc[0] / pre["shares"].iloc[0])
    ok = ratio_lo <= ratio <= ratio_hi
    lag = (post["filed"].iloc[0] - eff).days
    print(f"  {ticker} split eff {effective}: "
          f"{pre['shares'].iloc[0] / 1e9:.3f}B (filed {pre['filed'].iloc[0].date()}) -> "
          f"{post['shares'].iloc[0] / 1e9:.3f}B (filed {post['filed'].iloc[0].date()}, "
          f"eff+{lag}d), ratio {ratio:.2f} -> {'OK' if ok else 'FAIL'}")


def validation_report(df: pd.DataFrame, tickers: list[str]) -> None:
    print("\n" + "=" * 64)
    print("  SHARES OUTSTANDING PIT — VALIDATION")
    print("=" * 64)

    per = df.groupby("ticker").agg(rows=("shares", "size"), first_filed=("filed", "min"))
    print(f"\ncoverage: {len(per)}/{len(tickers)} tickers with facts, "
          f"{len(df):,} rows total")
    back_2011 = (per["first_filed"] <= "2011-12-31").sum()
    print(f"  history back to <=2011-12-31 (first filed): {back_2011} tickers")
    thin = per[per["rows"] < THIN_GROUPS].index.tolist()
    missing = [t for t in tickers if t not in per.index]
    print(f"  empty: {len(missing)}, thin (<{THIN_GROUPS} rows): {len(thin)}")
    for t in (missing + thin)[:15]:
        n = int(per.loc[t, "rows"]) if t in per.index else 0
        print(f"    {t:6s} {n} rows")
    by_concept = df.groupby("concept")["ticker"].nunique()
    print("  concept used (tickers):")
    for c, n in by_concept.items():
        print(f"    {c:42s} {n}")

    summed = sorted(df.loc[df["class_summed"], "ticker"].unique())
    print(f"\nmulti-class summed tickers ({len(summed)}): {summed}")

    print("\nsplit checks (share-count jump across effective date):")
    _split_check(df, "AAPL", "2014-06-09", 6.5, 7.5)   # 7:1
    _split_check(df, "AAPL", "2020-08-31", 3.6, 4.4)   # 4:1
    _split_check(df, "TSLA", "2020-08-31", 4.5, 5.5)   # 5:1

    aapl15 = df[(df["ticker"] == "AAPL") & (df["end"].dt.year == 2015)]["shares"]
    if len(aapl15):
        ok = aapl15.between(5.4e9, 6.1e9).all()
        print(f"  AAPL 2015 counts {aapl15.min() / 1e9:.2f}-{aapl15.max() / 1e9:.2f}B "
              f"(expect ~5.7-6.0B) -> {'OK' if ok else 'FAIL'}")
    msft = df[df["ticker"] == "MSFT"]["shares"]
    frac = msft.between(7.3e9, 7.8e9).mean()
    print(f"  MSFT rows in 7.3-7.8B: {frac * 100:.0f}% "
          f"(range {msft.min() / 1e9:.2f}-{msft.max() / 1e9:.2f}B; early years ~8.4B, "
          f"buybacks since) -> {'OK' if frac > 0.5 else 'FAIL'}")

    asof = pd.Timestamp("2023-06-30")
    if MARKET_DATA_PATH.exists():
        px = pd.read_parquet(MARKET_DATA_PATH, columns=["ticker", "date", "close"])
        px = px[px["date"] == asof]
        facts_by_ticker = dict(iter(df.groupby("ticker")))
        caps = {}
        for row in px.itertuples(index=False):
            facts = facts_by_ticker.get(row.ticker)
            if facts is not None:
                caps[row.ticker] = shares_asof(facts, asof) * row.close
        top = pd.Series(caps).dropna().sort_values(ascending=False)
        print(f"\nimplied market caps {asof.date()} (adjusted closes; tickers that "
              f"split later, e.g. NVDA 2024, are understated by design):")
        for t, v in top.head(8).items():
            print(f"    {t:6s} ${v / 1e12:.2f}T")
        top2 = set(top.head(2).index)
        print(f"  AAPL+MSFT are top 2 -> {'OK' if top2 == {'AAPL', 'MSFT'} else 'FAIL: ' + str(top2)}")
        tiny = top[top < 1e9]
        if len(tiny):
            print("  implied cap < $1B (share-unit mismatch suspects, e.g. BRK-B's "
                  "counts are Class-A-equivalent):")
            for t, v in tiny.items():
                print(f"    {t:6s} ${v / 1e6:,.0f}M")
    else:
        print(f"\n{MARKET_DATA_PATH.name} not found; skipping market-cap check")

    gaps = (df.drop_duplicates(["ticker", "filed"]).sort_values("filed")
            .groupby("ticker")["filed"].agg(lambda s: s.diff().dt.days.median()))
    gaps = gaps.dropna()
    print(f"\nstaleness: median gap between consecutive filed dates "
          f"(expect ~91d): median {gaps.median():.0f}d, p90 {gaps.quantile(0.9):.0f}d")
    slow = gaps[gaps > 200].sort_values(ascending=False)
    if len(slow):
        print(f"  tickers with median gap >200d: "
              f"{[(t, int(g)) for t, g in slow.items()]}")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch point-in-time shares outstanding from SEC XBRL")
    p.add_argument("--refresh", action="store_true",
                   help=f"Force re-fetch even if the parquet is <{MAX_AGE_DAYS:.0f} days old")
    args = p.parse_args()

    tickers = corpus_tickers()
    logger.info(f"Corpus: {len(tickers)} tickers under {FILINGS_DIR}")

    age_days = ((time.time() - SHARES_PIT_PATH.stat().st_mtime) / 86400
                if SHARES_PIT_PATH.exists() else None)
    if not args.refresh and age_days is not None and age_days < MAX_AGE_DAYS:
        logger.info(f"{SHARES_PIT_PATH.name} is {age_days:.1f} days old "
                    f"(<{MAX_AGE_DAYS:.0f}); skipping fetch (--refresh to force)")
        df = pd.read_parquet(SHARES_PIT_PATH)
    else:
        df = build_table(tickers)
        SHARES_PIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SHARES_PIT_PATH.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(SHARES_PIT_PATH)
        logger.success(f"Saved {len(df):,} rows -> {SHARES_PIT_PATH}")

    validation_report(df, tickers)


if __name__ == "__main__":
    main()
