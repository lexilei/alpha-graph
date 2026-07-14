"""Build data/cache/sc13_filings.parquet: SC 13D/13G filings ABOUT our panel companies.

Schedule 13D (>5% stake with control intent; statutory window 10 calendar days
pre-2024-02-05, 5 business days after) and Schedule 13G (the passive
counterpart) are filed by the ACQUIRER about the SUBJECT company, so they never
appear in our filings corpus (the companies' own 8-K/10-K/10-Q). This script is
DATA ACQUISITION ONLY for a future activist-stake factor: both 13D and 13G
(natural control group), with filer identity and amendment chains. No factor is
built here.

ROUTE (verified 2026-07-14 against three anchors — Starboard Value/DRI SC 13D
2013-12-23, Third Point/CPB SC 13D 2018-08-09, Icahn/OXY SC 13D 2020-03-12):
  * data.sec.gov/submissions/CIK<subject>.json DOES list SC 13D/G rows where
    the CIK is the SUBJECT (alongside rows where it is the FILER). Association
    sets are a SUPERSET of the browse-edgar ATOM company listing (CPB: 62 = 62;
    DRI: 119 vs 100 — ATOM omits 19 pre-2001 rows), and rows carry
    acceptanceDateTime and fileNumber. Primary route.
  * Role disambiguation: within CIK X's submissions, an SC 13 row carries the
    SUBJECT's Williams Act file number ("005-…") — non-blank exactly when X is
    the subject; filer-role rows are blank. Verified: BlackRock (CIK 1364742)
    shows 45,899 blank filer-role rows vs 58 own-subject rows; Berkshire 331 vs
    104; OXY's own 13D on Western Midstream (2019-08-19) is blank in OXY's
    spine; CPB's 13Ds on Snyder's-Lance (2017) / Sovos (2023) are blank. Without
    this filter, asset-manager members (BLK, BRK, …) would contaminate the
    panel with filings about OTHER companies.
  * Filer identity is NOT in the submissions rows, and browse-edgar ATOM titles
    do not carry it either (checked — titles are just the form name). The
    quarterly master.idx DOES: one row per associated party (subject + filer)
    per filing, with CIK and conformed name, so ~65 quarterly files resolve
    every filer without fetching ~40k filing headers. EDGAR full-text search
    also returns both parties (display_names) but pages 10 hits/request
    (~20k+ requests); unused.

FORM VOCABULARY: EDGAR renamed the form types on 2024-12-18 (the structured
13D/G mandate compliance date) from "SC 13D"/"SC 13G" to "SCHEDULE 13D"/
"SCHEDULE 13G" (same "/A" amendments) — a filter on the old names alone
silently ends the panel at 2024-12-16 (browse-edgar type=SC+13 has the same
blind spot). Both vocabularies are captured (in the submissions rows AND in
master.idx) and canonicalized to SC 13D / SC 13D/A / SC 13G / SC 13G/A.

CIK SUCCESSION: the submissions API is per-CIK, and several panel companies
changed CIK inside the window (holdco reorgs, redomiciliations, TPL's 2021
trust->corp conversion), so querying only the CURRENT CIK silently loses every
pre-reorg subject filing — TPL lost its only three activist 13D originals
(Horizon Kinetics 2016-03-08, SoftVest 2019-03-15, Santa Monica Partners
2019-04-08). OLD_CIK_MAP below carries the verified old CIKs; their spines are
fetched like any subject, tagged with the PANEL ticker, and merged through the
same subject-role filter and accession dedup. XOM needs no entry: the
filing_acceptance corpus already carries both its CIKs. Discovery route for
future additions: screen tickers whose earliest sc13 row starts years after
2010-07 despite an older company, name-match the company lineage in
data/cache/sc13_master_idx.parquet party names, then verify the candidate CIK's
submissions JSON (entity name / formerNames, in-window subject-role rows).

TIMEZONE: acceptanceDateTime is first handled as in fetch_acceptance_datetimes
.py (API UTC -> tz-naive US/Eastern wall time, row-local mislabeled-ET rules
A/B). The filer-level blanket rule C is NOT applied: it targeted companies
whose OWN transmissions are mislabeled, while SC 13 rows are transmitted by
third-party filers, so a subject-level blanket would overcorrect.

ACCEPTANCE RE-SOURCE (13D family): without rule C, mislabeled-ET rows whose
true acceptance is MIDDAY are undetectable — rules A/B only fire out-of-window
or against the dissemination calendar (verified: MSCI SC 13D 2012-11-28 SGML
header says 16:48:57 ET, the API-derived value was 11:48:57; GEN SC 13D/A
2019-08-15 was off by -4h). The 13D family is the tradeable event, so for
SC 13D + SC 13D/A rows (~2k) acceptance_ts is re-sourced from the
authoritative SGML header <ACCEPTANCE-DATETIME> (native ET wall time, no tz
inference) via a Range fetch of the first 2000 bytes of the full-submission
.txt (pattern of fetch_sic_history.py; some EDGAR nodes ignore Range and
reply 200, so the reader streams and hard-stops). The ~29k 13G-family rows
keep the API route with rules A/B; acceptance_source records the route.

NOT CAPTURED (verified absent from this route): the statutory event date
("Date of Event Which Requires Filing", the 5% crossing) — the submissions
reportDate field is empty for SC 13 rows. It lives on the filing cover page /
index header only; enriching the ~200 SC 13D originals would be a one-shot
follow-up fetch, deliberately not done for the ~28k-row panel.

Output (data/cache/sc13_filings.parquet), deduped on accession:
    subject_ticker  panel ticker of the SUBJECT company
    subject_cik     10-digit zero-padded CIK of the subject (str)
    accession       digits-only accession number (str)
    form            SC 13D / SC 13D/A / SC 13G / SC 13G/A (canonical)
    is_amendment    form ends with "/A"
    filing_date     official filing date (datetime, midnight)
    acceptance_ts   EDGAR acceptance datetime, tz-naive ET wall time (NaT if absent)
    acceptance_source  "sgml_header" (13D family, header fetched) or
                    "submissions_api" (13G family; 13D rows whose header fetch failed)
    filer_name      conformed name of the filing party (None if unresolved)
    filer_cik       10-digit zero-padded CIK of the filing party (None if unresolved)

Intermediate cache: data/cache/sc13_master_idx.parquet (SC 13 party rows from
quarterly master.idx files; immutable past quarters are never re-fetched).

Usage:
    python scripts/fetch_sc13_filings.py             # skip if cache <7 days old
    python scripts/fetch_sc13_filings.py --refresh   # force re-fetch
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import date

import pandas as pd
import requests
from loguru import logger

from alpha_graph.config import CACHE_DIR

OUT_PATH = CACHE_DIR / "sc13_filings.parquet"
IDX_CACHE_PATH = CACHE_DIR / "sc13_master_idx.parquet"
ACCEPTANCE_PATH = CACHE_DIR / "filing_acceptance.parquet"
SUBMISSIONS_BASE = "https://data.sec.gov/submissions/"
IDX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
USER_AGENT = "Lei Zhihan leizhihan0606@gmail.com"  # SEC fair-access identity
KEEP_FORMS = {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}
# EDGAR's renamed form types since 2024-12-18 (structured 13D/G mandate).
FORM_CANON = {"SCHEDULE 13D": "SC 13D", "SCHEDULE 13D/A": "SC 13D/A",
              "SCHEDULE 13G": "SC 13G", "SCHEDULE 13G/A": "SC 13G/A"}
RAW_FORMS = KEEP_FORMS | set(FORM_CANON)
MIN_DATE = "2010-07-01"  # panel event-history start
FIRST_QUARTER = (2010, 3)
REQUEST_INTERVAL = 0.17  # ~6 req/s, comfortably under SEC's 10 req/s ceiling
MAX_AGE_DAYS = 7.0

# Old CIKs of panel companies whose LISTED entity changed CIK inside the
# window. Each verified 2026-07-14 against its submissions JSON (entity name /
# formerNames, in-window subject-role SC 13 rows) after name-matching the
# master.idx party list. Look-alikes deliberately excluded: Viacom Inc 1339947
# (pre-2019 VIAB, a different listed lineage that merged INTO CBS=813828),
# Broadcom Corp 1054374 (BRCM, the company Avago ACQUIRED), BlackRock
# closed-end funds 832327/880280, T Rowe Price Associates 80255 and Duke
# Energy Carolinas 30371 (subsidiaries — their listed years pre-date the
# window), Google Ventures / CapitalG / Linden Capital / Ares Management LLC
# (filers, never our subjects).
OLD_CIK_MAP: dict[str, list[str]] = {
    "TPL": ["0000097517"],   # Texas Pacific Land Trust -> corp 1811074, 2021-01
    "BLK": ["0001364742"],   # BlackRock Inc. (now BlackRock Finance) -> holdco 2012383, 2024-10
    "PSKY": ["0000813828"],  # CBS Corp -> ViacomCBS -> Paramount Global -> Skydance holdco 2041610, 2025-08
    "DIS": ["0001001039"],   # Walt Disney Co (now TWDC Enterprises 18) -> holdco 1744489, 2019-03
    "GOOG": ["0001288776"],  # Google Inc. -> Alphabet 1652044, 2015-10
    "BG": ["0001144519"],    # Bunge Ltd -> Bunge Global SA 1996862, 2023-11
    "APA": ["0000006769"],   # Apache Corp -> APA Corp holdco 1841666, 2021-03
    "CI": ["0000701221"],    # Cigna Corp (now Cigna Holding) -> Cigna Group 1739940, 2018-12
    "MDT": ["0000064670"],   # Medtronic Inc -> Medtronic plc 1613103, 2015-01
    "AVGO": ["0001441634",   # Avago Technologies Ltd -> Broadcom Ltd, 2016-02
             "0001649338"],  # Broadcom Ltd (now Broadcom Pte.) -> Broadcom Inc 1730168, 2018-04
    "LIN": ["0000884905"],   # Praxair Inc (now Linde Inc) -> Linde plc 1707925, 2018-10
}

_last_request_t = 0.0


# --------------------------------------------------------------------------- #
# HTTP (rate-limited, retry once on 429/5xx) — conventions of
# fetch_acceptance_datetimes.py
# --------------------------------------------------------------------------- #

def _get(session: requests.Session, url: str, timeout: int = 60) -> requests.Response:
    global _last_request_t
    for attempt in (0, 1):
        wait = REQUEST_INTERVAL - (time.monotonic() - _last_request_t)
        if wait > 0:
            time.sleep(wait)
        _last_request_t = time.monotonic()
        try:
            resp = session.get(url, timeout=timeout)
        except requests.RequestException as e:
            if attempt == 1:
                raise
            logger.warning(f"{url}: {e}; retrying in 2s")
            time.sleep(2)
            continue
        if resp.status_code == 200:
            return resp
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
# Corpus: distinct subject CIK/ticker pairs from the acceptance table
# --------------------------------------------------------------------------- #

def load_corpus() -> pd.DataFrame:
    acc = pd.read_parquet(ACCEPTANCE_PATH, columns=["cik", "ticker"])
    corpus = acc.drop_duplicates().sort_values(["cik", "ticker"]).reset_index(drop=True)
    dupes = corpus[corpus.duplicated("cik", keep=False)]
    if len(dupes):  # one ticker per CIK: keep first alphabetically (precedent)
        logger.info(f"CIKs with several tickers, keeping first: {dupes.to_dict('records')}")
        corpus = corpus.drop_duplicates("cik", keep="first")
    return corpus


def augment_corpus_with_old_ciks(corpus: pd.DataFrame) -> pd.DataFrame:
    """corpus + one row per OLD_CIK_MAP entry, tagged with the PANEL ticker so
    pre-reorg subject filings land in the panel history. Rejects an old CIK
    that collides with a live corpus CIK and map tickers that left the panel
    (either would mean the map is stale). Pure; tested."""
    extra = pd.DataFrame(
        [{"cik": cik, "ticker": t} for t, ciks in OLD_CIK_MAP.items() for cik in ciks])
    unknown = sorted(set(OLD_CIK_MAP) - set(corpus["ticker"]))
    if unknown:
        raise ValueError(f"OLD_CIK_MAP tickers not in the corpus: {unknown}")
    out = pd.concat([corpus, extra], ignore_index=True)
    if not out["cik"].is_unique:
        dup = sorted(out.loc[out["cik"].duplicated(keep=False), "cik"].unique())
        raise ValueError(f"OLD_CIK_MAP collides with current corpus CIKs: {dup}")
    return out.sort_values(["cik", "ticker"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Submissions spine: SC 13 rows under each SUBJECT CIK
# --------------------------------------------------------------------------- #

def _parse_block(block: dict, cik: str, ticker: str) -> list[dict]:
    """Parse one parallel-array filings block, keeping only KEEP_FORMS rows."""
    accs = block.get("accessionNumber", [])
    forms = block.get("form", [])
    fdates = block.get("filingDate", [])
    adts = block.get("acceptanceDateTime", [])
    fnums = block.get("fileNumber", [])

    def at(arr: list, i: int) -> str | None:
        return arr[i] if i < len(arr) else None

    return [
        {
            "subject_cik": cik,
            "subject_ticker": ticker,
            "accession": accs[i],
            "form": forms[i],
            "filing_date": fdates[i],
            "acceptance_raw": at(adts, i),
            "file_number": at(fnums, i) or "",
        }
        for i in range(len(forms))
        if forms[i].strip() in RAW_FORMS
    ]


def fetch_spine(session: requests.Session, cik: str, ticker: str) -> list[dict]:
    """All SC 13 rows associated with `cik` (both roles; filtered later),
    from filings.recent plus archive pages that overlap MIN_DATE."""
    j = _get(session, f"{SUBMISSIONS_BASE}CIK{cik}.json").json()
    rows = _parse_block(j.get("filings", {}).get("recent", {}), cik, ticker)
    pages = sorted(j.get("filings", {}).get("files", []),
                   key=lambda p: p.get("filingTo", ""), reverse=True)
    for page in pages:
        if page.get("filingTo", "9999-99-99") < MIN_DATE:
            continue  # page ends before our window
        pj = _get(session, SUBMISSIONS_BASE + page["name"]).json()
        block = pj.get("filings", {}).get("recent", pj)  # archive pages are bare blocks
        rows.extend(_parse_block(block, cik, ticker))
    return rows


def fetch_all_spines(session: requests.Session, corpus: pd.DataFrame) -> pd.DataFrame:
    all_rows: list[dict] = []
    for i, r in enumerate(corpus.itertuples(index=False), 1):
        try:
            all_rows.extend(fetch_spine(session, r.cik, r.ticker))
        except Exception as e:  # noqa: BLE001
            logger.error(f"CIK {r.cik} ({r.ticker}): fetch failed: {e}")
        if i % 50 == 0:
            logger.info(f"{i}/{len(corpus)} subject CIKs fetched ({len(all_rows):,} raw rows)")
    return pd.DataFrame(all_rows)


# --------------------------------------------------------------------------- #
# Normalization (pure; tested)
# --------------------------------------------------------------------------- #

def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Form filter + canonicalization (post-2024-12 "SCHEDULE 13x" -> "SC 13x"),
    SUBJECT-role filter (non-blank Williams Act file number), window filter,
    key normalization, amendment flag, dedup on accession."""
    df = raw.copy()
    df["form"] = df["form"].str.strip().replace(FORM_CANON)
    df = df[df["form"].isin(KEEP_FORMS)]
    n_filer_role = int((df["file_number"].fillna("") == "").sum())
    df = df[df["file_number"].fillna("") != ""]  # keep SUBJECT-role rows only
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df[df["filing_date"] >= pd.Timestamp(MIN_DATE)]
    df["accession"] = df["accession"].str.replace(r"\D", "", regex=True)
    df["subject_cik"] = df["subject_cik"].astype(str).str.zfill(10)
    df["is_amendment"] = df["form"].str.endswith("/A")
    df = (df.sort_values(["accession", "subject_cik"], kind="stable")
            .drop_duplicates("accession", keep="first")
            .reset_index(drop=True))
    df.attrs["n_filer_role_dropped"] = n_filer_role
    return df


# --------------------------------------------------------------------------- #
# Quarterly master.idx: filer identity per accession
# --------------------------------------------------------------------------- #

def quarters_through(today: date) -> list[str]:
    """['2010Q3', ..., current quarter]."""
    y, q = FIRST_QUARTER
    out = []
    while (y, q) <= (today.year, (today.month - 1) // 3 + 1):
        out.append(f"{y}Q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def parse_master_idx(text: str, quarter: str) -> list[dict]:
    """SC 13 party rows from one master.idx: one row per (CIK, filing),
    forms canonicalized. Format: CIK|Company Name|Form Type|Date Filed|Filename."""
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        form = FORM_CANON.get(parts[2].strip(), parts[2].strip())
        if form not in KEEP_FORMS:
            continue
        acc = parts[4].rsplit("/", 1)[-1].removesuffix(".txt")
        rows.append({
            "quarter": quarter,
            "cik": parts[0].strip().zfill(10),
            "name": parts[1].strip(),
            "form": form,
            "date": parts[3].strip(),
            "accession": "".join(ch for ch in acc if ch.isdigit()),
        })
    return rows


def fetch_master_idx(session: requests.Session, today: date) -> pd.DataFrame:
    """SC 13 party rows for every quarter since FIRST_QUARTER, cached; past
    quarters are immutable, the current quarter is always re-fetched."""
    needed = quarters_through(today)
    cached = (pd.read_parquet(IDX_CACHE_PATH) if IDX_CACHE_PATH.exists()
              else pd.DataFrame(columns=["quarter", "cik", "name", "form", "date", "accession"]))
    to_fetch = [q for q in needed if q not in set(cached["quarter"])] + [needed[-1]]
    to_fetch = sorted(set(to_fetch))
    cached = cached[~cached["quarter"].isin(to_fetch)]

    parts = [cached]
    for i, quarter in enumerate(to_fetch, 1):
        y, q = int(quarter[:4]), int(quarter[-1])
        try:
            text = _get(session, IDX_URL.format(year=y, q=q)).content.decode("latin-1")
        except Exception as e:  # noqa: BLE001
            logger.error(f"master.idx {quarter}: fetch failed: {e}")
            continue
        parts.append(pd.DataFrame(parse_master_idx(text, quarter)))
        if i % 10 == 0:
            logger.info(f"{i}/{len(to_fetch)} master.idx quarters fetched")
    idx = pd.concat(parts, ignore_index=True)

    IDX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = IDX_CACHE_PATH.with_suffix(".parquet.tmp")
    idx.to_parquet(tmp, index=False)
    tmp.replace(IDX_CACHE_PATH)
    logger.info(f"master.idx cache: {len(idx):,} SC 13 party rows "
                f"({len(to_fetch)} quarters fetched, {cached['quarter'].nunique()} from cache)")
    return idx


def attach_filers(spine: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """filer = idx parties on the accession minus the subject CIK; if several
    (joint filings), the lowest CIK, deterministically. Pure; tested."""
    parties = idx[idx["accession"].isin(spine["accession"])][["accession", "cik", "name"]]
    merged = spine[["accession", "subject_cik"]].merge(parties, on="accession", how="left")
    filers = (merged[merged["cik"].notna() & (merged["cik"] != merged["subject_cik"])]
              .sort_values(["accession", "cik"], kind="stable"))
    agg = filers.groupby("accession").agg(filer_cik=("cik", "first"),
                                          filer_name=("name", "first"),
                                          n_filers=("cik", "nunique"))
    out = spine.merge(agg, on="accession", how="left")
    out["n_filers"] = out["n_filers"].fillna(0).astype(int)
    return out


# --------------------------------------------------------------------------- #
# Acceptance timestamps: UTC -> ET wall time (precedent rules A/B, no C),
# then the 13D family re-sourced from the authoritative SGML header
# --------------------------------------------------------------------------- #

SGML_HEAD_BYTES = 2000  # <ACCEPTANCE-DATETIME> sits in the first ~150 bytes
_ACCEPTANCE_RE = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")


def parse_sgml_acceptance(text: str) -> pd.Timestamp | None:
    """<ACCEPTANCE-DATETIME>YYYYMMDDHHMMSS from an SGML header head — native
    ET wall time, no timezone inference. None when absent or malformed."""
    m = _ACCEPTANCE_RE.search(text)
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S")
    except ValueError:
        return None


def fetch_sgml_head(session: requests.Session, cik: str, accession: str) -> str | None:
    """First SGML_HEAD_BYTES of the full-submission .txt via a Range request.

    Some EDGAR nodes ignore Range and reply 200 with the whole document, so
    the reader streams and hard-stops regardless of status. Tries the subject
    CIK, then (on 404) the accession-prefix CIK — the first 10 digits of an
    accession are the transmitting filer, which always has the file. Retries
    once on 429/5xx/connection errors; None on permanent failure.
    """
    global _last_request_t
    dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    cik_candidates = [str(int(cik))]
    prefix = str(int(accession[:10]))
    if prefix not in cik_candidates:
        cik_candidates.append(prefix)

    for ck in cik_candidates:
        url = f"{ARCHIVES_BASE}/{ck}/{accession}/{dashed}.txt"
        for attempt in (0, 1):
            wait = REQUEST_INTERVAL - (time.monotonic() - _last_request_t)
            if wait > 0:
                time.sleep(wait)
            _last_request_t = time.monotonic()
            try:
                resp = session.get(
                    url,
                    headers={"Range": f"bytes=0-{SGML_HEAD_BYTES - 1}",
                             "Accept-Encoding": "identity"},
                    stream=True, timeout=30,
                )
            except requests.RequestException:
                if attempt == 0:
                    time.sleep(2)
                    continue
                break
            try:
                if resp.status_code in (200, 206):
                    buf = b""
                    for chunk in resp.iter_content(chunk_size=2048):
                        buf += chunk
                        if len(buf) >= SGML_HEAD_BYTES:
                            break
                    return buf.decode("latin-1", errors="replace")
                if resp.status_code == 404:
                    break  # permanent for this CIK; try the prefix CIK
                if attempt == 0 and (resp.status_code == 429 or resp.status_code >= 500):
                    time.sleep(float(resp.headers.get("Retry-After", 5)))
                    continue
                break
            finally:
                resp.close()
    return None


def apply_sgml_acceptance(df: pd.DataFrame, header_ts: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Overwrite acceptance_ts/acceptance_source from fetched SGML header
    values (keyed by digits-only accession). Rows without a header value keep
    the submissions-API value and source. Pure; tested."""
    out = df.copy()
    ts = out["accession"].map(header_ts)
    hit = ts.notna()
    out.loc[hit, "acceptance_ts"] = ts[hit]
    out.loc[hit, "acceptance_source"] = "sgml_header"
    return out


def resource_13d_acceptance(session: requests.Session, df: pd.DataFrame) -> pd.DataFrame:
    """Re-source acceptance_ts for every SC 13D / SC 13D/A row from its SGML
    header (see module docstring). Requires an acceptance_source column."""
    fam = df[df["form"].str.startswith("SC 13D")]
    logger.info(f"SGML acceptance pass: {len(fam):,} 13D-family rows "
                f"(~{len(fam) * REQUEST_INTERVAL / 60:.0f} min at the rate gate)")
    header_ts: dict[str, pd.Timestamp] = {}
    failures = 0
    for i, r in enumerate(fam.itertuples(index=False), 1):
        text = fetch_sgml_head(session, r.subject_cik, r.accession)
        ts = parse_sgml_acceptance(text) if text is not None else None
        if ts is None:
            failures += 1
            logger.warning(f"{r.subject_ticker} {r.accession}: no SGML acceptance; "
                           "keeping submissions_api value")
        else:
            header_ts[r.accession] = ts
        if i % 250 == 0:
            logger.info(f"{i:,}/{len(fam):,} SGML headers fetched ({failures} failures)")
    return apply_sgml_acceptance(df, header_ts)

def to_eastern(raw: pd.Series, filing_date: pd.Series) -> tuple[pd.Series, pd.Series]:
    """API UTC -> tz-naive ET wall time; row-local mislabeled-ET corrections
    (rules A/B of fetch_acceptance_datetimes.py). Returns (acceptance_ts, fixed)."""
    raw = raw.astype("string")
    utc_et = (pd.to_datetime(raw, utc=True, errors="coerce", format="mixed")
              .dt.tz_convert("America/New_York").dt.tz_localize(None))
    wall = pd.to_datetime(raw.str.replace(r"[Zz]|\.\d+", "", regex=True), errors="coerce")

    day_min = utc_et.dt.hour * 60 + utc_et.dt.minute
    wall_min = wall.dt.hour * 60 + wall.dt.minute
    wall_in_window = wall_min.between(6 * 60, 22 * 60 + 5)
    rule_a = ~day_min.between(6 * 60, 22 * 60 + 5) & wall_in_window
    date_gap = (filing_date - utc_et.dt.normalize()).dt.days
    rule_b = ((day_min < 17 * 60 + 30) & date_gap.between(1, 4)
              & (wall_min >= 17 * 60 + 20) & wall_in_window)
    fix = ((rule_a | rule_b) & utc_et.notna()).fillna(False).astype(bool)
    return utc_et.where(~fix, wall), fix


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

COLUMNS = ["subject_ticker", "subject_cik", "accession", "form", "is_amendment",
           "filing_date", "acceptance_ts", "acceptance_source", "filer_name", "filer_cik"]


def build_table(corpus: pd.DataFrame) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})

    corpus = augment_corpus_with_old_ciks(corpus)
    n_old = sum(len(v) for v in OLD_CIK_MAP.values())
    logger.info(f"Corpus augmented with {n_old} pre-reorg CIKs "
                f"for {len(OLD_CIK_MAP)} succession tickers")
    raw = fetch_all_spines(session, corpus)
    logger.info(f"raw SC 13 associations fetched: {len(raw):,}")
    spine = normalize(raw)
    logger.info(f"subject-role rows in window: {len(spine):,} "
                f"(dropped {spine.attrs['n_filer_role_dropped']:,} filer-role rows)")

    idx = fetch_master_idx(session, date.today())
    df = attach_filers(spine, idx)
    n_multi = int((df["n_filers"] > 1).sum())
    logger.info(f"filer resolved: {df['filer_cik'].notna().mean() * 100:.2f}% "
                f"({n_multi} joint-filer submissions, lowest CIK kept)")

    df["acceptance_ts"], fixed = to_eastern(df["acceptance_raw"], df["filing_date"])
    logger.info(f"acceptance_ts non-null: {df['acceptance_ts'].notna().mean() * 100:.2f}%; "
                f"tz rules A/B corrected {int(fixed.sum())} rows")

    df["acceptance_source"] = "submissions_api"
    api_ts = df["acceptance_ts"].copy()
    df = resource_13d_acceptance(session, df)
    delta_h = (df["acceptance_ts"] - api_ts).dt.total_seconds() / 3600
    n_sgml = int((df["acceptance_source"] == "sgml_header").sum())
    logger.info(f"SGML re-source: {n_sgml:,} 13D-family rows on header authority; "
                f"{int((delta_h.abs() >= 1).sum())} moved >=1h vs the API route")
    return df[COLUMNS]


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #

ANCHORS = [
    # (ticker, form, filing_date, filer substring, note)
    ("DRI", "SC 13D", "2013-12-23", "STARBOARD", "Starboard Value on Darden"),
    ("CPB", "SC 13D", "2018-08-09", "THIRD POINT", "Third Point on Campbell Soup"),
    ("OXY", "SC 13D", "2020-03-12", "ICAHN",
     "Icahn on Occidental (crossed 5% in 2020-03; his 2019 campaign stayed <5%, "
     "and 2019's only OXY-associated 13D is OXY-as-filer on Western Midstream)"),
]

TS_ANCHORS = [
    # (label, accession, expected acceptance_ts) — SGML headers read by hand
    # 2026-07-14; both are API-mislabeled midday-shift rows rules A/B cannot see.
    ("MSCI SC 13D 2012-11-28 (ValueAct)", "000141881212000079", "2012-11-28 16:48:57"),
    ("GEN SC 13D/A 2019-08-15 (Starboard)", "000092189519002251", "2019-08-15 16:52:31"),
]


def validation_report(df: pd.DataFrame, corpus: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("  SC 13D/13G FILINGS — VALIDATION")
    print("=" * 72)
    print(f"\nrows: {len(df):,}   subjects: {df['subject_ticker'].nunique()} "
          f"of {corpus['ticker'].nunique()} corpus companies   "
          f"span {df['filing_date'].min().date()} -> {df['filing_date'].max().date()}")
    print(f"acceptance_ts non-null: {df['acceptance_ts'].notna().mean() * 100:.2f}%   "
          f"filer resolved: {df['filer_cik'].notna().mean() * 100:.2f}%")
    fam_13d = df["form"].str.startswith("SC 13D")
    n_sgml = int((df["acceptance_source"] == "sgml_header").sum())
    print(f"acceptance_source: sgml_header {n_sgml:,} of {int(fam_13d.sum()):,} "
          f"13D-family rows ({n_sgml / max(int(fam_13d.sum()), 1) * 100:.2f}%); "
          f"13G family stays submissions_api")
    old_ciks = {c for cs in OLD_CIK_MAP.values() for c in cs}
    per_old = df[df["subject_cik"].isin(old_ciks)].groupby("subject_ticker").size()
    print(f"pre-reorg rows via OLD_CIK_MAP: {int(per_old.sum()):,} across "
          f"{len(per_old)}/{len(OLD_CIK_MAP)} succession tickers  "
          + "  ".join(f"{t}:{n}" for t, n in per_old.items()))
    unresolved = df[df["filer_cik"].isna()]
    if len(unresolved):
        print(f"  unresolved-filer sample (of {len(unresolved)}):")
        for _, r in unresolved.head(5).iterrows():
            print(f"    {r.subject_ticker:6s} {r.form:9s} {r.filing_date.date()} {r.accession}")

    print("\ncounts by form by year:")
    pivot = (df.assign(year=df["filing_date"].dt.year)
             .pivot_table(index="year", columns="form", aggfunc="size", fill_value=0))
    print(pivot.to_string())

    print("\nanchor verification:")
    for ticker, form, fdate, filer_sub, note in ANCHORS:
        hit = df[(df["subject_ticker"] == ticker) & (df["form"] == form)
                 & (df["filing_date"] == pd.Timestamp(fdate))
                 & df["filer_name"].str.upper().str.contains(filer_sub, na=False)]
        status = "OK " if len(hit) else "MISS"
        detail = (f"filer={hit.iloc[0]['filer_name']} acc={hit.iloc[0]['accession']}"
                  if len(hit) else "not found")
        print(f"  [{status}] {ticker} {form} {fdate} — {note}\n         {detail}")
    for label, acc, expected in TS_ANCHORS:
        hit = df[df["accession"] == acc]
        got = str(hit.iloc[0]["acceptance_ts"]) if len(hit) else "row missing"
        status = "OK " if got == expected else "MISS"
        print(f"  [{status}] {label}: acceptance {got} (expect {expected})")

    d13 = df[df["form"] == "SC 13D"]
    breadth = d13["subject_ticker"].nunique()
    print(f"\nBREADTH CEILING: {breadth}/{corpus['ticker'].nunique()} subjects "
          f"with >=1 original SC 13D ({d13.shape[0]:,} originals)")
    n_fam = df[df["form"].str.startswith("SC 13D")]["subject_ticker"].nunique()
    print(f"  incl. amendments: {n_fam} subjects touched by the 13D family")
    per_year = d13.groupby(d13["filing_date"].dt.year)["subject_ticker"].nunique()
    print("  distinct subjects with a new original 13D per year:")
    print("   " + "  ".join(f"{y}:{n}" for y, n in per_year.items()))

    print("\ntop-20 filer names on SC 13D originals (case-folded):")
    top = d13["filer_name"].str.upper().value_counts().head(20)
    for name, n in top.items():
        print(f"  {n:3d}  {name}")

    fam = df[df["filer_cik"].notna()].assign(
        family=df["form"].str.extract(r"SC (13[DG])", expand=False))
    d_pairs = fam[fam["family"] == "13D"].groupby(["subject_ticker", "filer_cik"])
    stats = d_pairs["is_amendment"].agg(["size", "sum"])
    stats["orig"] = stats["size"] - stats["sum"]
    print(f"\n13D amendment chains (subject-filer pairs, filer resolved): {len(stats):,} pairs")
    print(f"  pairs with an in-window original: {(stats['orig'] > 0).sum():,}; "
          f"orphan /A-only pairs (original pre-2010-07 or missed): {(stats['orig'] == 0).sum():,}")
    with_orig = stats[stats["orig"] > 0]
    if len(with_orig):
        print(f"  amendments per original-bearing pair: mean {with_orig['sum'].mean():.1f}, "
              f"median {with_orig['sum'].median():.0f}, max {int(with_orig['sum'].max())}")

    grp = fam.groupby(["subject_ticker", "filer_cik"])
    both = grp["family"].agg(set)
    both = both[both.apply(lambda s: {"13D", "13G"} <= s)].index
    stand_down, escalation = [], []
    for key in both:
        g = grp.get_group(key)
        d_dates = g.loc[g["family"] == "13D", "filing_date"]
        g_dates = g.loc[g["family"] == "13G", "filing_date"]
        if (g_dates > d_dates.max()).any():
            stand_down.append((key, d_dates.max(), g_dates[g_dates > d_dates.max()].min(),
                               g["filer_name"].iloc[0]))
        if (d_dates > g_dates.min()).any() and g_dates.min() < d_dates.min():
            escalation.append(key)
    print(f"\n13D -> 13G stand-downs (same subject-filer, 13G after last 13D): {len(stand_down)}")
    for (tick, _), last_d, first_g, fname in stand_down[:5]:
        print(f"    {tick:6s} {fname[:40]:40s} last 13D {last_d.date()} -> 13G {first_g.date()}")
    print(f"13G -> 13D escalations (passive first, then active): {len(escalation)}")

    mins = d13["acceptance_ts"].dt.hour * 60 + d13["acceptance_ts"].dt.minute
    print(f"\nsharpness — SC 13D originals (n={len(d13):,}). The tradeable event is the")
    print("  EDGAR acceptance; the statutory 5%-crossing date (10 calendar days before")
    print("  filing pre-2024-02-05, 5 business days after) is only on the cover page")
    print("  (submissions reportDate is empty for SC 13 rows; one-shot follow-up).")
    print(f"  accepted at/after 16:00 ET (not tradeable at same-day close): "
          f"{(mins >= 16 * 60).mean() * 100:.1f}%")
    print(f"  accepted at/after 17:30 ET (disseminated next business day):  "
          f"{(mins >= 17 * 60 + 30).mean() * 100:.1f}%")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Fetch SC 13D/13G filings about panel companies")
    p.add_argument("--refresh", action="store_true",
                   help=f"Force re-fetch even if the parquet is <{MAX_AGE_DAYS:.0f} days old")
    args = p.parse_args()

    corpus = load_corpus()
    logger.info(f"Corpus: {len(corpus)} subject CIKs")

    age_days = ((time.time() - OUT_PATH.stat().st_mtime) / 86400 if OUT_PATH.exists() else None)
    if not args.refresh and age_days is not None and age_days < MAX_AGE_DAYS:
        logger.info(f"{OUT_PATH.name} is {age_days:.1f} days old (<{MAX_AGE_DAYS:.0f}); "
                    "skipping fetch (--refresh to force)")
        table = pd.read_parquet(OUT_PATH)
    else:
        table = build_table(corpus)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_PATH.with_suffix(".parquet.tmp")
        table.to_parquet(tmp, index=False)
        tmp.replace(OUT_PATH)
        logger.success(f"Saved {len(table):,} rows -> {OUT_PATH}")

    validation_report(table, corpus)


if __name__ == "__main__":
    main()
