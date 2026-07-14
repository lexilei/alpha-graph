"""Build data/cache/sic_history.parquet: point-in-time SIC per 10-K, from EDGAR headers.

The panel's `sector` column is a CURRENT GICS-style snapshot (sector_map.parquet),
i.e. mildly non-PIT. This script builds the PIT alternative: the SEC Standard
Industrial Classification each company carried AS OF each 10-K filing date, read
from the EDGAR submission header of the filing itself. 10-K only (annual
resolution is enough for an industry map; 10-Qs would triple the request count).

Source: https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{acc_dashed}.txt
whose first ~2-4 KB hold the <SEC-HEADER> block with, per FILER,
    STANDARD INDUSTRIAL CLASSIFICATION:  <NAME> [<CODE>]     (modern)
    STANDARD INDUSTRIAL CLASSIFICATION:  <CODE>              (early-1990s, bare)
We request only the head via `Range: bytes=0-7999`. EDGAR Archives honors Range
(HTTP 206) most of the time, but occasionally a node replies 200 with the full
document (observed on a 31 MB 10-K, 2026-07-13), so the reader streams and
hard-stops after HEADER_BYTES regardless of status. Multi-filer headers
(co-registrant subsidiaries) repeat the SIC line: we take the SIC of the FILER
block whose CENTRAL INDEX KEY matches the row's CIK, falling back to the first
SIC line. Fair access: UA identity below, <=8 req/s across threads, retry once
on 429/5xx/connection errors, log-and-continue on hard failures (they are NOT
written to the parquet, so the next incremental run retries them).

Output columns (one row per fetched 10-K accession):
    cik          10-digit zero-padded CIK (str, as in filing_acceptance.parquet)
    ticker       corpus ticker
    accession    digits-only accession number (str)
    filing_date  official 10-K filing date (datetime)
    sic_code     4-digit zero-padded SIC (str); NA = header fetched but had no
                 SIC line (kept so incremental runs don't refetch; the loader
                 alpha_graph.data.sic_pit drops NA rows on load)
    sic_name     SIC label from the header (NA on bare-code early-1990s rows)

Usage:
    python scripts/fetch_sic_history.py              # incremental: only new accessions
    python scripts/fetch_sic_history.py --refresh    # full refetch
    python scripts/fetch_sic_history.py --limit 50   # smoke test on 50 accessions
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from loguru import logger

from alpha_graph.config import CACHE_DIR

ACCEPTANCE_PATH = CACHE_DIR / "filing_acceptance.parquet"
SECTOR_MAP_PATH = CACHE_DIR / "sector_map.parquet"
OUT_PATH = CACHE_DIR / "sic_history.parquet"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
USER_AGENT = "Lei Zhihan leizhihan0606@gmail.com"  # SEC fair-access identity
REQUEST_INTERVAL = 0.13  # ~8 req/s across all threads; SEC ceiling is 10
WORKERS = 6              # concurrency hides latency; the gate enforces the rate
HEADER_BYTES = 8000      # SEC-HEADER incl. several FILER blocks fits well within
CROSSTAB_DATE = "2024-06-30"
MIX_DATES = ("2011-06-30", "2024-06-30")

_rate_lock = threading.Lock()
_last_request_t = 0.0


class FetchError(RuntimeError):
    """Permanent failure for one accession (after retry / CIK fallback)."""


# --------------------------------------------------------------------------- #
# HTTP (global token gate, Range + streamed hard-stop, retry once)
# --------------------------------------------------------------------------- #

def _gate() -> None:
    """Serialize request STARTS so all workers together stay <=8 req/s."""
    global _last_request_t
    with _rate_lock:
        wait = REQUEST_INTERVAL - (time.monotonic() - _last_request_t)
        if wait > 0:
            time.sleep(wait)
        _last_request_t = time.monotonic()


def fetch_header_text(session: requests.Session, cik: str, accession: str) -> str:
    """First HEADER_BYTES of the full-submission .txt for one accession.

    Tries the row's CIK, then (on 404) the accession-prefix CIK — the first 10
    digits of an accession are the transmitting filer, which differs from the
    submissions-API CIK for ticker re-points (e.g. holdco reorgs).
    """
    dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    cik_candidates = [str(int(cik))]
    prefix = str(int(accession[:10]))
    if prefix not in cik_candidates:
        cik_candidates.append(prefix)

    last_err: object = "unknown"
    for ck in cik_candidates:
        url = f"{ARCHIVES_BASE}/{ck}/{accession}/{dashed}.txt"
        for attempt in (0, 1):
            _gate()
            try:
                resp = session.get(
                    url,
                    headers={"Range": f"bytes=0-{HEADER_BYTES - 1}",
                             "Accept-Encoding": "identity"},
                    stream=True, timeout=30,
                )
            except requests.RequestException as e:
                last_err = e
                if attempt == 0:
                    time.sleep(2)
                    continue
                break
            try:
                if resp.status_code in (200, 206):
                    # 200 = node ignored Range: stop after HEADER_BYTES anyway.
                    buf = b""
                    for chunk in resp.iter_content(chunk_size=4096):
                        buf += chunk
                        if len(buf) >= HEADER_BYTES:
                            break
                    return buf.decode("utf-8", errors="replace")
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code == 404:
                    break  # permanent for this CIK; try the fallback CIK
                if attempt == 0 and (resp.status_code == 429 or resp.status_code >= 500):
                    time.sleep(float(resp.headers.get("Retry-After", 5)))
                    continue
                break
            finally:
                resp.close()
    raise FetchError(f"{dashed}: {last_err}")


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #

_SIC_LINE = re.compile(r"STANDARD INDUSTRIAL CLASSIFICATION:([^\r\n]*)", re.I)
_CIK_LINE = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)", re.I)
_NAME_CODE = re.compile(r"^(.*?)\s*\[(\d{1,4})\]$")


def parse_sic(text: str, cik: str) -> tuple[str, str | None] | None:
    """(sic_code, sic_name) for the FILER block matching `cik`, else the first.

    Handles both header formats: modern "NAME [CODE]" and early-1990s bare
    "CODE" (no name). Returns None when no SIC line parses.
    """
    sics: list[tuple[int, str, str | None]] = []
    for m in _SIC_LINE.finditer(text):
        tail = m.group(1).strip()
        nc = _NAME_CODE.match(tail)
        if nc:
            code, name = nc.group(2), (nc.group(1).strip() or None)
        elif tail.isdigit():
            code, name = tail, None
        else:
            continue
        sics.append((m.start(), code.zfill(4), name))
    if not sics:
        return None

    # In each FILER block the CENTRAL INDEX KEY line precedes the SIC line:
    # a SIC belongs to the closest CIK above it.
    target = int(cik)
    cik_positions = [(m.start(), int(m.group(1))) for m in _CIK_LINE.finditer(text)]
    for pos, code, name in sics:
        owners = [c for p, c in cik_positions if p < pos]
        if owners and owners[-1] == target:
            return code, name
    return sics[0][1], sics[0][2]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def load_corpus() -> pd.DataFrame:
    """All 10-K rows of the acceptance table (accession, cik, ticker, filing_date)."""
    df = pd.read_parquet(
        ACCEPTANCE_PATH, columns=["accession", "cik", "ticker", "form", "filing_date"])
    k = df[df["form"] == "10-K"].drop(columns="form").reset_index(drop=True)
    logger.info(f"Corpus: {len(k):,} 10-K accessions, {k['ticker'].nunique()} tickers, "
                f"{k['filing_date'].min().date()} -> {k['filing_date'].max().date()}")
    return k


def _fetch_one(session: requests.Session, row: pd.Series) -> dict:
    text = fetch_header_text(session, row["cik"], row["accession"])
    parsed = parse_sic(text, row["cik"])
    if parsed is None:
        code, name = None, None
    else:
        code, name = parsed
    return {"cik": row["cik"], "ticker": row["ticker"], "accession": row["accession"],
            "filing_date": row["filing_date"], "sic_code": code, "sic_name": name}


def build_rows(todo: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Fetch+parse every accession in `todo`; returns (rows, failures)."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows: list[dict] = []
    failures: list[dict] = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_fetch_one, session, row): row for _, row in todo.iterrows()}
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001 — log-and-continue by design
                failures.append({"ticker": row["ticker"], "accession": row["accession"],
                                 "filing_date": row["filing_date"], "error": str(e)})
                logger.warning(f"{row['ticker']} {row['accession']}: {e}")
            if i % 500 == 0:
                rate = i / (time.monotonic() - t0)
                eta = (len(futures) - i) / rate / 60
                logger.info(f"{i:,}/{len(futures):,} fetched "
                            f"({rate:.1f} req/s, ~{eta:.0f} min left, "
                            f"{len(failures)} failures)")
    return rows, failures


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #

def validation_report(table: pd.DataFrame, corpus: pd.DataFrame) -> None:
    # Lazy: the loader ships with this pipeline; fetching works without it.
    from alpha_graph.data.sic_pit import sic_asof, sic_division

    print("\n" + "=" * 72)
    print("  SIC HISTORY (PIT) — VALIDATION")
    print("=" * 72)

    ok = table.dropna(subset=["sic_code"])
    n_corpus = corpus["accession"].nunique()
    print(f"\naccession coverage: {len(ok):,}/{n_corpus:,} 10-Ks with a parsed SIC "
          f"({len(ok) / n_corpus * 100:.2f}%)")
    tickers = sorted(corpus["ticker"].unique())
    covered = set(ok["ticker"])
    missing = [t for t in tickers if t not in covered]
    print(f"ticker coverage: {len(covered)}/{len(tickers)} corpus tickers with >=1 SIC "
          f"({len(covered) / len(tickers) * 100:.2f}%)")
    if missing:
        print(f"  tickers with NO SIC: {missing[:20]}")

    # --- stability: who ever changes 2-digit major group across their 10-Ks --
    h = ok.sort_values(["ticker", "filing_date"], kind="mergesort").copy()
    h["s2"] = h["sic_code"].str[:2]
    grp = h.groupby("ticker")
    h["prev_s2"] = grp["s2"].shift()
    h["prev_code"] = grp["sic_code"].shift()
    h["prev_date"] = grp["filing_date"].shift()
    chg = h[h["prev_s2"].notna() & (h["s2"] != h["prev_s2"])]
    changers = chg["ticker"].unique()
    print(f"\nsic2 stability: {len(changers)}/{len(covered)} tickers ever change "
          f"2-digit major group across their 10-Ks ({len(chg)} change events)")
    print("  sample changers (up to 10 tickers):")
    for t in changers[:10]:
        for _, r in chg[chg["ticker"] == t].iterrows():
            print(f"    {t:6s} {r.prev_code} ({sic_division(r.prev_code)}) "
                  f"@{r.prev_date.date()} -> {r.sic_code} ({sic_division(r.sic_code)}) "
                  f"@{r.filing_date.date()}")

    # --- cross-tab: current-GICS snapshot vs PIT SIC division ---------------
    asof_codes = {t: sic_asof(sub, CROSSTAB_DATE) for t, sub in h.groupby("ticker")}
    if SECTOR_MAP_PATH.exists():
        sm = pd.read_parquet(SECTOR_MAP_PATH)
        sm = sm[sm["ticker"].isin(covered)].copy()
        sm["division"] = [sic_division(asof_codes.get(t)) for t in sm["ticker"]]
        known = sm[sm["division"] != "UNKNOWN"]
        print(f"\ncurrent-GICS sector vs SIC division as of {CROSSTAB_DATE} "
              f"(n={len(known)} tickers; {len(sm) - len(known)} without a 10-K by then):")
        ct = pd.crosstab(known["sector"], known["division"])
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(ct.to_string())
        modal = ct.idxmax(axis=1)
        off = [(s, d, int(ct.at[s, d])) for s in ct.index for d in ct.columns
               if d != modal[s] and ct.at[s, d] > 0]
        off.sort(key=lambda x: -x[2])
        print("\n  5 biggest off-diagonal cells (off = not the sector's modal division):")
        for s, d, n in off[:5]:
            ex = known[(known["sector"] == s) & (known["division"] == d)]["ticker"]
            print(f"    {s:22s} -> {d:22s} {n:3d}  e.g. {', '.join(ex.head(4))}")
    else:
        print("\n(sector_map.parquet missing — skipping GICS cross-tab)")

    # --- division mix drift --------------------------------------------------
    print("\ndivision mix (share of tickers with a SIC as of date):")
    mixes = {}
    for d in MIX_DATES:
        codes = {t: sic_asof(sub, d) for t, sub in h.groupby("ticker")}
        divs = pd.Series([sic_division(c) for c in codes.values()])
        divs = divs[divs != "UNKNOWN"]
        mixes[d] = divs.value_counts(normalize=True) * 100
    mix = pd.DataFrame(mixes).fillna(0.0).sort_values(MIX_DATES[-1], ascending=False)
    for div, r in mix.iterrows():
        print(f"    {div:22s} " + "  ".join(f"{r[d]:5.1f}%" for d in MIX_DATES))
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Fetch PIT SIC codes from EDGAR 10-K headers")
    p.add_argument("--refresh", action="store_true",
                   help="Refetch everything (default: only accessions not yet cached)")
    p.add_argument("--limit", type=int, default=None,
                   help="Fetch at most N accessions (smoke test)")
    args = p.parse_args()

    corpus = load_corpus()
    existing = None
    if OUT_PATH.exists() and not args.refresh:
        existing = pd.read_parquet(OUT_PATH)
        existing = existing[existing["accession"].isin(set(corpus["accession"]))]
        logger.info(f"Incremental: {len(existing):,} accessions already cached")

    todo = corpus if existing is None else corpus[~corpus["accession"].isin(
        set(existing["accession"]))]
    if args.limit is not None:
        todo = todo.head(args.limit)

    if len(todo):
        logger.info(f"Fetching {len(todo):,} headers "
                    f"(~{len(todo) * REQUEST_INTERVAL / 60:.0f} min at the rate gate)")
        rows, failures = build_rows(todo)
        new = pd.DataFrame(rows, columns=["cik", "ticker", "accession",
                                          "filing_date", "sic_code", "sic_name"])
        table = new if existing is None else pd.concat([existing, new], ignore_index=True)
        table = table.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_PATH.with_suffix(".parquet.tmp")
        table.to_parquet(tmp, index=False)
        tmp.replace(OUT_PATH)
        no_sic = int(table["sic_code"].isna().sum())
        logger.success(f"Saved {len(table):,} rows -> {OUT_PATH} "
                       f"({no_sic} without a SIC line, {len(failures)} fetch failures)")
        if failures:
            print(f"\nfetch failures ({len(failures)}, not cached — retried next run):")
            for f in failures[:20]:
                print(f"    {f['ticker']:6s} {f['filing_date'].date()} "
                      f"{f['accession']}  {f['error']}")
    else:
        table = existing
        logger.info("Nothing to fetch — cache already covers the corpus")

    validation_report(table, corpus)


if __name__ == "__main__":
    main()
