"""Extract SEC-mandated principal-customer disclosures from 10-K filings.

Segment-reporting rules (ASC 280, previously SFAS 131/SFAS 14) require a
public company to disclose any customer accounting for 10% or more of
consolidated revenue. Cohen & Frazzini (2008) built customer momentum on
exactly these disclosures (via Compustat segment files). This module
extracts the same disclosures from the 10-K Business-section texts the
graph pipeline already uses (same loader, same 20k-char truncation as
relationships.py, so evidence verifies against the exact text the LLM saw).

Differences from relationships.py (the v0 LLM graph):
- Extraction targets ONLY explicitly named principal/major customers,
  with their disclosed revenue share when stated.
- Verbatim evidence is ENFORCED at extraction time: an edge whose
  evidence quote (or customer name) is not an exact whitespace-normalized,
  case-insensitive substring of the text the LLM saw is REJECTED from the
  edge set and counted. (relationships.py only flags.) Rejected rows stay
  in the per-filing JSON caches with their flags for auditability.
- Customer names are free text, entity-resolved to panel tickers after
  extraction; non-panel customers (U.S. government, foreign OEMs) are
  KEPT with customer_ticker=None and tradeable=False — they matter for
  disclosure-completeness stats but are out of the tradeable edge set.
- Each edge carries a validity window: valid_from = filing_date,
  valid_to = filing date of the supplier's next 10-K that does NOT
  re-disclose that customer (NaT = still open). A 10-K that failed the
  regex prefilter is a known non-disclosure (it lacks concentration
  language) and closes windows; a matched-but-unprocessed 10-K is
  unknown and is skipped, never closing a window.

Pipeline:
  1. prefilter — regex for customer-concentration language (no LLM)
  2. cost gate — report matched count + projected LLM calls, then select
  3. extract  — LLM (same client/config as relationships.py) on selected
                filings; per-filing JSON cache; verbatim rejection
  4. build   — entity resolution, validity windows, CIK join, parquet

Usage:
    python -m alpha_graph.data.principal_customers --prefilter-only
    python -m alpha_graph.data.principal_customers [--workers 8]
    python -m alpha_graph.data.principal_customers --build-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR, cfg
from alpha_graph.data.relationships import (
    _evidence_in_text,
    _load_business_text,
    _normalize_ws,
)
from alpha_graph.utils.llm import extract_json

PRINCIPAL_CACHE_DIR = CACHE_DIR / "principal_customers"
PREFILTER_PATH = CACHE_DIR / "principal_customers_prefilter.parquet"
OUT_PATH = CACHE_DIR / "principal_customers.parquet"
ACCEPTANCE_PATH = CACHE_DIR / "filing_acceptance.parquet"

# LLM-call budget for the cost gate (see select_for_extraction).
DEFAULT_BUDGET = 3500

# ---------------------------------------------------------------------------
# 1. Prefilter — customer-concentration language (cheap, no LLM)
# ---------------------------------------------------------------------------

PREFILTER_PATTERNS = [
    # -- core customer-concentration language --
    r"principal\s+customers?",
    r"major\s+customers?",
    r"largest\s+customers?",
    r"accounted\s+for\s+(?:approximately\s+|about\s+)?\d+(?:\.\d+)?%",
    r"10%\s+(?:or\s+more\s+)?of\s+(?:our\s+|the\s+company['’]s\s+)?"
    r"(?:consolidated\s+|total\s+)?(?:net\s+)?(?:revenues?|sales)",
    r"concentration\s+of\s+credit\s+risk",
    # -- recall extensions (hand-check on SWKS showed the core set misses
    #    'Customer Concentration' headers and multi-year percent lists like
    #    'Apple ... was 66%, 65% and 63% of net revenue') --
    r"customer\s+concentration",
    r"significant\s+customers?",
    r"\d+(?:\.\d+)?%\s+(?:or\s+more\s+)?of\s+(?:[\w()'’-]+\s+){0,4}"
    r"(?:revenues?|sales|accounts\s+receivable)",
    r"ten\s+percent\s+(?:or\s+more\s+)?of",
]
CONCENTRATION_RE = re.compile(
    "|".join(f"(?:{p})" for p in PREFILTER_PATTERNS), re.IGNORECASE
)


def _amendment_accessions() -> set[str]:
    """Accessions that are really 10-K/A amendments (per SEC submissions).

    The downloader saved amendments under 10-K_*.json filenames; a Part III
    amendment carries no Business text and must be excluded — otherwise it
    would falsely close validity windows months after the real 10-K.
    """
    if not ACCEPTANCE_PATH.exists():
        logger.warning(f"{ACCEPTANCE_PATH} missing — cannot exclude 10-K/A "
                       f"amendments")
        return set()
    acc = pd.read_parquet(ACCEPTANCE_PATH, columns=["accession", "form"])
    return set(acc.loc[acc["form"] != "10-K", "accession"])


def prefilter_filings(
    tickers: list[str] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Scan every true 10-K's Business-section text for concentration language.

    Uses the same loader (and 20k-char truncation) as the LLM extraction,
    so the prefilter sees exactly the text the model would see. Filings
    that are really 10-K/A amendments are excluded.

    Returns a DataFrame over ALL true 10-K filings (matched and unmatched —
    the unmatched ones are needed later as known non-disclosures for
    validity windows): supplier_ticker, accession, filing_date,
    company_name, path, matched. Cached to PREFILTER_PATH.
    """
    full_scan = tickers is None
    if PREFILTER_PATH.exists() and not refresh and full_scan:
        df = pd.read_parquet(PREFILTER_PATH)
        logger.info(f"Prefilter loaded from cache: {len(df)} filings, "
                    f"{int(df['matched'].sum())} matched")
        return df

    if full_scan:
        tickers = sorted(
            d.name for d in FILINGS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    amendments = _amendment_accessions()
    n_amend = 0
    rows = []
    for i, ticker in enumerate(tickers, 1):
        for path in sorted((FILINGS_DIR / ticker).glob("10-K_*.json")):
            if path.stem.split("_")[2] in amendments:
                n_amend += 1
                continue
            try:
                text, company_name, filing_date = _load_business_text(path)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                logger.warning(f"prefilter: cannot read {path}: {e}")
                continue
            rows.append({
                "supplier_ticker": ticker,
                "accession": path.stem.split("_")[2],
                "filing_date": filing_date,
                "company_name": company_name,
                "path": str(path),
                "matched": bool(CONCENTRATION_RE.search(text)),
            })
        if i % 100 == 0:
            logger.info(f"  prefilter: {i}/{len(tickers)} tickers scanned")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df = df.sort_values(["supplier_ticker", "filing_date"]).reset_index(drop=True)
    if full_scan:  # never overwrite the full cache with a --tickers subset
        PREFILTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PREFILTER_PATH, index=False)
    logger.info(f"Prefilter: {int(df['matched'].sum())}/{len(df)} 10-K filings "
                f"match concentration language ({n_amend} 10-K/A amendments "
                f"excluded)")
    return df


# ---------------------------------------------------------------------------
# 2. Cost gate
# ---------------------------------------------------------------------------

def select_for_extraction(
    pre: pd.DataFrame, budget: int = DEFAULT_BUDGET
) -> tuple[pd.DataFrame, str]:
    """Pick the filings to send to the LLM, respecting the call budget.

    If matched <= budget: extract all matched filings. Otherwise: all
    filings from tickers that EVER match, plus the `budget` most-recent
    filings of never-matching tickers.
    """
    matched = pre[pre["matched"]]
    if len(matched) <= budget:
        note = (f"cost gate: {len(matched)} matched filings <= budget {budget} "
                f"-> extracting all matched filings ({len(matched)} LLM calls)")
        return matched.copy(), note
    ever = set(matched["supplier_ticker"])
    from_ever = pre[pre["supplier_ticker"].isin(ever)]
    others = (pre[~pre["supplier_ticker"].isin(ever)]
              .sort_values("filing_date").tail(budget))
    sel = pd.concat([from_ever, others]).sort_values(
        ["supplier_ticker", "filing_date"])
    note = (f"cost gate: {len(matched)} matched filings > budget {budget} "
            f"-> taking all {len(from_ever)} filings from {len(ever)} "
            f"ever-matching tickers plus the {len(others)} most-recent filings "
            f"of never-matching tickers ({len(sel)} LLM calls)")
    return sel, note


# ---------------------------------------------------------------------------
# 3. LLM extraction with verbatim-evidence enforcement
# ---------------------------------------------------------------------------

PRINCIPAL_CUSTOMER_PROMPT = """\
You are extracting regulation-mandated customer-concentration disclosures from an SEC 10-K filing. Segment-reporting rules (ASC 280 / SFAS 131) require a company to disclose any customer that accounts for 10% or more of its revenue, and many filings also name principal customers voluntarily.

## Reporting company (the supplier): {ticker} ({company_name})

## Filing text
<filing_text>
{business_text}
</filing_text>

List every customer of {ticker} that the filing text above EXPLICITLY NAMES as a principal, major, largest, or significant customer, or for which it discloses a share of revenue or sales. Respond with JSON:
{{
    "customers": [
        {{
            "customer_name": "...",
            "revenue_pct": 26.0,
            "evidence": "..."
        }}
    ]
}}

Rules:
- customer_name: the customer's name exactly as written in the filing text (verbatim), e.g. "Wal-Mart Stores, Inc." or "the U.S. Government". A customer must be a NAMED organization; never anonymous groups such as "three customers" or "our ten largest customers".
- revenue_pct: the disclosed percentage of {ticker}'s revenue / net sales attributable to that customer, as a plain number (26 for "26%"). If several years are disclosed, use the most recent. If only an accounts-receivable percentage is stated, or no percentage at all, use null.
- evidence: an exact, verbatim quote from the filing text above (at most 300 characters) containing the disclosure — ideally the sentence that names the customer and its revenue share. Copy it character-for-character: it MUST be an exact substring of the filing text. Do not paraphrase, shorten, fix typos, or change punctuation.
- Include ONLY customers — entities that BUY {ticker}'s goods or services. Do not include suppliers, partners, competitors, or companies mentioned in any other role.
- Do not use outside knowledge; extract only what this text states.
- List each customer at most once. If the filing names no specific customer, return {{"customers": []}}.
"""


def _coerce_pct(value) -> float | None:
    """Parse a revenue percentage into a float in (0, 100], else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    else:
        m = re.search(r"\d+(?:\.\d+)?", str(value))
        if not m:
            return None
        v = float(m.group(0))
    if not math.isfinite(v) or v <= 0 or v > 100:
        return None
    return v


def _validate_customers(
    raw: list, business_text: str
) -> tuple[list[dict], int, int]:
    """Normalize raw LLM customer dicts and verify evidence verbatim.

    Every row with a non-empty customer_name is returned, carrying
    `evidence_verified` and `name_verified` flags (both computed with the
    whitespace-normalized, case-insensitive substring check from
    relationships.py against the exact text the LLM saw). The edge set
    keeps only rows with BOTH flags True — callers must filter; the flags
    stay on the row so JSON caches remain auditable.

    Returns (rows, n_evidence_rejected, n_name_rejected) where the two
    counts partition the failing rows: evidence failure counts first, a
    name failure is counted only when the evidence verified.
    """
    rows: list[dict] = []
    n_evid = n_name = 0
    for c in raw or []:
        if not isinstance(c, dict):
            continue
        name = c.get("customer_name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        evidence = c.get("evidence", "")
        if not isinstance(evidence, str):
            evidence = ""
        ev_ok = _evidence_in_text(evidence, business_text)
        nm_ok = _evidence_in_text(name, business_text)
        if not ev_ok:
            n_evid += 1
        elif not nm_ok:
            n_name += 1
        rows.append({
            "customer_name": name,
            "revenue_pct": _coerce_pct(c.get("revenue_pct")),
            "evidence": evidence,
            "evidence_verified": bool(ev_ok),
            "name_verified": bool(nm_ok),
        })
    return rows, n_evid, n_name


def extract_for_filing(ticker: str, filing_path: Path, cache: bool = True) -> dict:
    """Extract principal customers from one 10-K via LLM, verbatim-enforced.

    Per-filing JSON cache (resume-safe). Parse/API failures raise (nothing
    cached) so a rerun retries them.
    """
    from openai import OpenAI

    cache_dir = PRINCIPAL_CACHE_DIR / ticker
    cache_path = cache_dir / f"{filing_path.stem}.json"
    if cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    business_text, company_name, filing_date = _load_business_text(filing_path)
    accession = filing_path.stem.split("_")[2]

    def _finish(customers: list[dict], n_evid: int = 0, n_name: int = 0) -> dict:
        result = {
            "ticker": ticker,
            "company_name": company_name,
            "filing_date": filing_date,
            "accession": accession,
            "model": cfg.llm_model,
            "n_rejected_evidence": n_evid,
            "n_rejected_name": n_name,
            "customers": customers,
        }
        if cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    if len(business_text) < 200:
        logger.debug(f"[{ticker}] filing too short, skipping: {filing_path.name}")
        return _finish([])

    if not cfg.llm_api_key:
        raise ValueError("LLM_API_KEY not set. Add it to .env")

    client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url or None,
                    timeout=180)
    prompt = PRINCIPAL_CUSTOMER_PROMPT.format(
        ticker=ticker, company_name=company_name, business_text=business_text)
    response = client.chat.completions.create(
        model=cfg.llm_model,
        temperature=0.0,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    # Raises on unparseable output -> no cache write -> retried on rerun.
    parsed = extract_json(response.choices[0].message.content)
    raw = parsed.get("customers", [])
    if not isinstance(raw, list):
        raw = []
    customers, n_evid, n_name = _validate_customers(raw, business_text)
    n_kept = sum(1 for c in customers
                 if c["evidence_verified"] and c["name_verified"])
    logger.info(f"[{ticker}] {filing_date}: {n_kept} kept, "
                f"{n_evid + n_name} rejected (verbatim)")
    return _finish(customers, n_evid, n_name)


def extract_all(
    selected: pd.DataFrame, workers: int = 8, passes: int = 2
) -> dict:
    """Run LLM extraction over the selected filings (threaded, resumable).

    Returns counters: n_selected, n_cached_prior, n_calls, n_failed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = [(r.supplier_ticker, Path(r.path)) for r in selected.itertuples()]
    n_cached_prior = sum(
        1 for t, p in jobs
        if (PRINCIPAL_CACHE_DIR / t / f"{p.stem}.json").exists()
    )
    n_calls = 0
    n_failed = 0
    for pass_i in range(passes):
        todo = [(t, p) for t, p in jobs
                if not (PRINCIPAL_CACHE_DIR / t / f"{p.stem}.json").exists()]
        if not todo:
            break
        logger.info(f"extract pass {pass_i + 1}: {len(todo)} filings, "
                    f"{workers} workers, model={cfg.llm_model}")
        n_failed = 0
        n_done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(extract_for_filing, t, p): (t, p)
                       for t, p in todo}
            for fut in as_completed(futures):
                t, p = futures[fut]
                n_done += 1
                try:
                    fut.result()
                    n_calls += 1
                except Exception as e:
                    n_failed += 1
                    logger.error(f"[{t}] {p.name} failed: {e}")
                if n_done % 100 == 0:
                    logger.info(f"  progress: {n_done}/{len(todo)} "
                                f"({n_failed} failed)")
    return {"n_selected": len(jobs), "n_cached_prior": n_cached_prior,
            "n_calls": n_calls, "n_failed": n_failed}


# ---------------------------------------------------------------------------
# 4. Entity resolution: customer_name -> panel ticker
# ---------------------------------------------------------------------------

# Legal-suffix tokens stripped (iteratively) from the right end of a name.
_SUFFIX_TOKENS = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "COMPANIES", "LTD", "LIMITED", "PLC", "LLC", "LLP", "LP",
    "NV", "SA", "AG", "SE", "&",
}

# Hand alias map (raw form -> panel ticker); normalized at map build time
# and applied only when the target ticker is in the panel.
HAND_ALIASES = {
    "Walmart": "WMT", "Wal-Mart": "WMT", "Wal-Mart Stores": "WMT",
    "Walmart Stores": "WMT", "Sam's Club": "WMT",
    "AT&T": "T", "AT & T": "T", "AT&T Mobility": "T", "SBC Communications": "T",
    "Hewlett-Packard": "HPQ", "HP": "HPQ",
    "Hewlett Packard Enterprise": "HPE", "HPE": "HPE",
    "Apple Computer": "AAPL", "Apple": "AAPL",
    "Ford": "F", "Ford Motor": "F",
    "GM": "GM", "General Motors": "GM",
    "Boeing": "BA",
    "Amazon": "AMZN", "Amazon.com": "AMZN", "Amazon Web Services": "AMZN",
    "AWS": "AMZN",
    "Google": "GOOGL", "Alphabet": "GOOGL", "YouTube": "GOOGL",
    "Facebook": "META", "Meta": "META", "Meta Platforms": "META",
    "Verizon": "VZ", "Verizon Wireless": "VZ", "Verizon Communications": "VZ",
    "Cellco Partnership": "VZ",
    "T-Mobile": "TMUS", "T-Mobile US": "TMUS", "T-Mobile USA": "TMUS",
    "Sprint": "TMUS",  # post-2020 merger; pre-2020 Sprint is not the panel line
    "McKesson": "MCK", "AmerisourceBergen": "COR", "Cencora": "COR",
    "Cardinal Health": "CAH",
    "CVS": "CVS", "CVS Caremark": "CVS", "CVS Health": "CVS",
    "CVS Pharmacy": "CVS",
    "UnitedHealthcare": "UNH", "United Healthcare": "UNH",
    "UnitedHealth": "UNH", "UnitedHealth Group": "UNH", "Optum": "UNH",
    "OptumRx": "UNH",
    "Anthem": "ELV", "Elevance Health": "ELV", "WellPoint": "ELV",
    "Raytheon": "RTX", "Raytheon Technologies": "RTX",
    "United Technologies": "RTX",
    "Lockheed": "LMT", "Lockheed Martin": "LMT",
    "GE": "GE", "General Electric": "GE",
    "IBM": "IBM", "International Business Machines": "IBM",
    "P&G": "PG", "Procter & Gamble": "PG", "Procter and Gamble": "PG",
    "Coca-Cola": "KO", "PepsiCo": "PEP", "Pepsi": "PEP",
    "Costco": "COST", "Costco Wholesale": "COST",
    "Home Depot": "HD", "Lowe's": "LOW", "Lowes": "LOW",
    "Target": "TGT", "Kroger": "KR", "Best Buy": "BBY", "Sysco": "SYY",
    "Cisco": "CSCO", "Cisco Systems": "CSCO",
    "Dell": "DELL", "Dell Technologies": "DELL",
    "Intel": "INTC", "Microsoft": "MSFT", "Qualcomm": "QCOM",
    "Broadcom": "AVGO", "Micron": "MU", "Texas Instruments": "TXN",
    "NVIDIA": "NVDA",
    "Exxon Mobil": "XOM", "ExxonMobil": "XOM", "Chevron": "CVX",
    "JPMorgan": "JPM", "JP Morgan": "JPM", "JPMorgan Chase": "JPM",
    "Citibank": "C", "Citigroup": "C",
    "Goldman Sachs": "GS", "Morgan Stanley": "MS",
    "Wells Fargo": "WFC", "Bank of America": "BAC",
    "American Express": "AXP",
    "FedEx": "FDX", "UPS": "UPS", "United Parcel Service": "UPS",
    "Caterpillar": "CAT", "John Deere": "DE", "Deere": "DE",
    "Disney": "DIS", "Walt Disney": "DIS",
    "Delta Air Lines": "DAL", "Delta Airlines": "DAL",
    "United Airlines": "UAL", "Comcast": "CMCSA",
    "Netflix": "NFLX", "Tesla": "TSLA",
    "McDonald's": "MCD", "Starbucks": "SBUX", "Nike": "NKE",
    "Johnson & Johnson": "JNJ", "Pfizer": "PFE", "Merck": "MRK",
    "General Dynamics": "GD", "Northrop Grumman": "NOC",
}


def _norm_name(name: str) -> str:
    """Normalize a company name for matching.

    Uppercase; unify unicode punctuation; drop periods/commas/apostrophes;
    hyphens and slashes become spaces; collapse whitespace; strip leading
    "THE" and trailing legal suffixes (Inc/Corp/Co/Ltd/...).
    """
    s = unicodedata.normalize("NFKC", name)
    s = (s.replace("’", "'").replace("‘", "'")
         .replace("“", '"').replace("”", '"'))
    s = s.upper()
    s = s.replace(".", "").replace(",", " ").replace("'", "").replace('"', " ")
    s = re.sub(r"[-–—/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split(" ")
    while len(toks) > 1 and toks[-1] in _SUFFIX_TOKENS:
        toks.pop()
    while len(toks) > 1 and toks[0] == "THE":
        toks.pop(0)
    out = " ".join(toks)
    return out if out else name.upper().strip()


def build_alias_map(names_by_ticker: dict[str, list[str]]) -> dict[str, str]:
    """Build normalized-name -> panel-ticker map.

    `names_by_ticker`: panel ticker -> all company_name strings seen across
    its filings (covers renames like Facebook->Meta automatically). Hand
    aliases are applied last (they win), and only for panel tickers.
    On auto-name collisions the alphabetically-first ticker wins (logged).
    """
    amap: dict[str, str] = {}
    collisions = []
    for ticker in sorted(names_by_ticker):
        for name in names_by_ticker[ticker]:
            if not isinstance(name, str) or not name.strip():
                continue
            key = _norm_name(name)
            if key in amap and amap[key] != ticker:
                collisions.append((key, amap[key], ticker))
                continue
            amap[key] = ticker
    if collisions:
        logger.debug(f"alias map: {len(collisions)} auto-name collisions "
                     f"(first ticker kept), e.g. {collisions[:5]}")
    panel = set(names_by_ticker)
    for raw, ticker in HAND_ALIASES.items():
        if ticker in panel:
            amap[_norm_name(raw)] = ticker
    return amap


def resolve_customer(name: str, alias_map: dict[str, str]) -> str | None:
    """Map a raw customer name to a panel ticker, or None if non-panel."""
    return alias_map.get(_norm_name(name))


def _customer_key(name: str, ticker: str | None) -> str:
    """Identity used for dedup and re-disclosure matching across years."""
    return ticker if ticker else "n:" + _norm_name(name)


# ---------------------------------------------------------------------------
# 5. Validity windows
# ---------------------------------------------------------------------------

def compute_validity(edges: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Attach valid_from / valid_to to each edge.

    edges: supplier_ticker, filing_date, customer_key (plus anything else).
    inventory: one row per supplier 10-K: supplier_ticker, filing_date,
    known (bool — do we know its disclosed set?), keys (iterable of
    customer_keys disclosed there; empty for known non-disclosures).

    An edge's window is [filing_date, first LATER known filing by the same
    supplier whose key set does NOT contain the customer). Later filings
    that re-disclose the customer extend the window past themselves;
    unknown filings are skipped (they never close a window); no closing
    filing -> valid_to = NaT (still open).
    """
    inv: dict[str, dict[pd.Timestamp, tuple[bool, frozenset]]] = {}
    for r in inventory.itertuples():
        d = pd.Timestamp(r.filing_date)
        inv.setdefault(r.supplier_ticker, {})
        by_date = inv[r.supplier_ticker]
        known, keys = by_date.get(d, (False, frozenset()))
        # Same-date filings merge: known if any known; union of keys.
        by_date[d] = (known or bool(r.known),
                      keys | frozenset(r.keys if r.known else ()))
    inv_sorted = {
        t: sorted((d, k, s) for d, (k, s) in by_date.items())
        for t, by_date in inv.items()
    }

    valid_to = []
    for r in edges.itertuples():
        d0 = pd.Timestamp(r.filing_date)
        vt = pd.NaT
        for d, known, keys in inv_sorted.get(r.supplier_ticker, []):
            if d <= d0 or not known:
                continue
            if r.customer_key in keys:
                continue  # re-disclosed -> window extends past this filing
            vt = d
            break
        valid_to.append(vt)
    out = edges.copy()
    out["valid_from"] = pd.to_datetime(out["filing_date"])
    out["valid_to"] = pd.to_datetime(pd.Series(valid_to, index=out.index))
    return out


def finalize_edges(
    edges: pd.DataFrame,
    alias_map: dict[str, str],
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve customers, dedupe per filing, add windows + tradeable flag.

    edges: one row per verified extraction — supplier_ticker, accession,
    filing_date, customer_name, revenue_pct, evidence, model (evidence
    already verbatim-verified upstream).

    Non-panel customers keep customer_ticker=None and tradeable=False;
    a supplier naming itself is also flagged non-tradeable.
    """
    df = edges.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["customer_ticker"] = [
        resolve_customer(n, alias_map) for n in df["customer_name"]
    ]
    df["customer_key"] = [
        _customer_key(n, t)
        for n, t in zip(df["customer_name"], df["customer_ticker"])
    ]
    # One edge per (filing, customer): prefer a stated revenue_pct (larger
    # first), then longer evidence; deterministic.
    df["_pct_sort"] = df["revenue_pct"].fillna(-1.0)
    df["_ev_len"] = df["evidence"].str.len()
    df = (df.sort_values(["_pct_sort", "_ev_len", "customer_name"],
                         ascending=[False, False, True], kind="stable")
            .drop_duplicates(subset=["supplier_ticker", "accession",
                                     "customer_key"], keep="first")
            .drop(columns=["_pct_sort", "_ev_len"]))
    df = compute_validity(df, inventory)
    df["tradeable"] = (
        df["customer_ticker"].notna()
        & (df["customer_ticker"] != df["supplier_ticker"])
    )
    df = df.rename(columns={"customer_name": "customer_name_raw"})
    return df.sort_values(["supplier_ticker", "filing_date",
                           "customer_key"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Build parquet from per-filing caches
# ---------------------------------------------------------------------------

def _load_caches(pre: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Read per-filing JSON caches; split kept vs rejected rows.

    Returns (kept_edges_df, stats). Kept = evidence_verified AND
    name_verified; both flags kept as columns (all True by construction).
    """
    rows = []
    n_raw = n_rej_evid = n_rej_name = n_cached = 0
    for r in pre.itertuples():
        cache_path = (PRINCIPAL_CACHE_DIR / r.supplier_ticker
                      / f"{Path(r.path).stem}.json")
        if not cache_path.exists():
            continue
        n_cached += 1
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        for c in data.get("customers", []):
            n_raw += 1
            ok = c.get("evidence_verified") and c.get("name_verified")
            if not c.get("evidence_verified"):
                n_rej_evid += 1
            elif not c.get("name_verified"):
                n_rej_name += 1
            if not ok:
                continue
            rows.append({
                "supplier_ticker": r.supplier_ticker,
                "accession": data.get("accession", r.accession),
                "filing_date": data.get("filing_date", r.filing_date),
                "customer_name": c["customer_name"],
                "revenue_pct": c.get("revenue_pct"),
                "evidence": c.get("evidence", ""),
                "evidence_verified": True,
                "name_verified": True,
                "model": data.get("model", ""),
            })
    stats = {"n_filings_cached": n_cached, "n_raw": n_raw,
             "n_rejected_evidence": n_rej_evid, "n_rejected_name": n_rej_name,
             "n_kept": len(rows)}
    return pd.DataFrame(rows), stats


def build_principal_customers(
    pre: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the final parquet from prefilter inventory + LLM caches."""
    if pre is None:
        pre = prefilter_filings()

    kept, stats = _load_caches(pre)
    if kept.empty:
        logger.warning("No verified principal-customer edges found in caches")
        return kept, stats

    # Entity resolution from panel company names (all names ever seen).
    names_by_ticker = {
        t: sorted(set(g["company_name"].dropna()))
        for t, g in pre.groupby("supplier_ticker")
    }
    alias_map = build_alias_map(names_by_ticker)

    # Inventory for validity windows: every 10-K on disk.
    #   cached          -> known, keys = kept keys of that filing
    #   no cache, unmatched -> known non-disclosure (prefilter found no
    #                          concentration language), keys = {}
    #   no cache, matched   -> unknown (not processed / failed)
    kept["customer_ticker_tmp"] = [
        resolve_customer(n, alias_map) for n in kept["customer_name"]
    ]
    kept["customer_key_tmp"] = [
        _customer_key(n, t)
        for n, t in zip(kept["customer_name"], kept["customer_ticker_tmp"])
    ]
    keys_by_accession: dict[str, set] = {}
    for r in kept.itertuples():
        keys_by_accession.setdefault(r.accession, set()).add(r.customer_key_tmp)
    kept = kept.drop(columns=["customer_ticker_tmp", "customer_key_tmp"])

    inv_rows = []
    n_unknown = 0
    for r in pre.itertuples():
        cache_exists = (PRINCIPAL_CACHE_DIR / r.supplier_ticker
                        / f"{Path(r.path).stem}.json").exists()
        known = cache_exists or not r.matched
        if not known:
            n_unknown += 1
        inv_rows.append({
            "supplier_ticker": r.supplier_ticker,
            "filing_date": r.filing_date,
            "known": known,
            "keys": tuple(keys_by_accession.get(r.accession, ())),
        })
    inventory = pd.DataFrame(inv_rows)
    if n_unknown:
        logger.warning(f"validity windows: {n_unknown} matched-but-unprocessed "
                       f"filings treated as unknown (never close a window)")

    df = finalize_edges(kept, alias_map, inventory)

    # CIK join from the acceptance parquet (accession, no dashes).
    if ACCEPTANCE_PATH.exists():
        acc = (pd.read_parquet(ACCEPTANCE_PATH, columns=["accession", "cik"])
               .drop_duplicates("accession"))
        df = df.merge(acc.rename(columns={"cik": "supplier_cik"}),
                      on="accession", how="left")
    else:
        logger.warning(f"{ACCEPTANCE_PATH} missing — supplier_cik left null")
        df["supplier_cik"] = None

    cols = ["supplier_ticker", "supplier_cik", "accession", "filing_date",
            "customer_name_raw", "customer_ticker", "revenue_pct", "evidence",
            "evidence_verified", "model", "valid_from", "valid_to",
            "customer_key", "tradeable"]
    df = df[cols]
    assert df["evidence_verified"].all(), "kept rows must all be verified"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info(f"Saved {len(df)} principal-customer edges to {OUT_PATH}")
    stats["n_final"] = len(df)
    stats["n_unknown_inventory"] = n_unknown
    return df, stats


# ---------------------------------------------------------------------------
# 7. Validation report
# ---------------------------------------------------------------------------

def validation_report(df: pd.DataFrame, stats: dict, pre: pd.DataFrame) -> None:
    """Print the extraction-quality numbers the confirmation needs."""
    n_raw = stats["n_raw"]
    n_rej = stats["n_rejected_evidence"] + stats["n_rejected_name"]
    print("\n=== principal_customers validation ===")
    print(f"prefilter: {int(pre['matched'].sum())}/{len(pre)} 10-K filings "
          f"matched concentration language")
    print(f"filings with LLM cache: {stats['n_filings_cached']}")
    print(f"raw LLM edges: {n_raw}")
    print(f"verbatim-rejected: {n_rej} "
          f"({stats['n_rejected_evidence']} evidence, "
          f"{stats['n_rejected_name']} name) = "
          f"{n_rej / n_raw:.1%} of raw" if n_raw else "no raw edges")
    print(f"kept (verbatim-verified) rows: {stats['n_kept']}; "
          f"after per-filing dedup: {len(df)}")
    if df.empty:
        return
    print(f"edges with revenue_pct: {int(df['revenue_pct'].notna().sum())} "
          f"({df['revenue_pct'].notna().mean():.1%})")
    print(f"distinct suppliers with >=1 edge: {df['supplier_ticker'].nunique()}")
    n_panel = int(df["customer_ticker"].notna().sum())
    print(f"panel-resolvable customer share: {n_panel}/{len(df)} "
          f"({n_panel / len(df):.1%}); tradeable: {int(df['tradeable'].sum())}")

    print("\ntop-10 most-named customers (by supplier-filing edges):")
    top = (df.groupby("customer_key")
             .agg(n=("customer_key", "size"),
                  ticker=("customer_ticker", "first"),
                  example=("customer_name_raw", "first"))
             .sort_values("n", ascending=False).head(10))
    for k, r in top.iterrows():
        tk = r["ticker"] if pd.notna(r["ticker"]) else "-"
        print(f"  {r['n']:5d}  [{tk:>5}]  {r['example']}")

    closed = df[df["valid_to"].notna()]
    if len(closed):
        life = (closed["valid_to"] - closed["valid_from"]).dt.days
        print(f"\nvalidity windows: {len(closed)}/{len(df)} closed "
              f"({len(closed) / len(df):.1%}); median lifetime "
              f"{life.median():.0f} days (p25 {life.quantile(.25):.0f}, "
              f"p75 {life.quantile(.75):.0f})")

    # Hand-check 1: who names Apple as a principal customer?
    aapl = df[(df["customer_ticker"] == "AAPL")
              & (df["supplier_ticker"] != "AAPL")]
    print(f"\nhand-check 1 — suppliers naming Apple: "
          f"{sorted(aapl['supplier_ticker'].unique())}")
    for _, r in (aapl.sort_values("revenue_pct", ascending=False).head(3)
                 .iterrows()):
        print(f"  {r['supplier_ticker']} {r['filing_date'].date()} "
              f"pct={r['revenue_pct']}: {r['evidence'][:120]!r}")

    # Hand-check 2: consumer names naming Walmart.
    wmt = df[(df["customer_ticker"] == "WMT")
             & (df["supplier_ticker"] != "WMT")]
    print(f"\nhand-check 2 — suppliers naming Walmart: "
          f"{sorted(wmt['supplier_ticker'].unique())}")
    for _, r in (wmt.sort_values("revenue_pct", ascending=False).head(3)
                 .iterrows()):
        print(f"  {r['supplier_ticker']} {r['filing_date'].date()} "
              f"pct={r['revenue_pct']}: {r['evidence'][:120]!r}")

    # Hand-check 3: re-open one edge's raw filing and confirm verbatim.
    probe = (wmt if len(wmt) else df).iloc[0]
    fpath = pre.loc[pre["accession"] == probe["accession"], "path"]
    if len(fpath):
        text, _, _ = _load_business_text(Path(fpath.iloc[0]))
        ok = _evidence_in_text(probe["evidence"], text)
        print(f"\nhand-check 3 — evidence vs raw filing "
              f"({probe['supplier_ticker']} {probe['accession']}): "
              f"verbatim={ok}")
        print(f"  quote: {probe['evidence'][:200]!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SEC-disclosed principal customers from 10-Ks")
    parser.add_argument("--tickers", nargs="+", help="subset (debug)")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent LLM calls")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help="LLM call budget for the cost gate")
    parser.add_argument("--prefilter-only", action="store_true",
                        help="report prefilter + cost gate, no LLM")
    parser.add_argument("--build-only", action="store_true",
                        help="rebuild parquet from existing caches")
    parser.add_argument("--refresh-prefilter", action="store_true")
    args = parser.parse_args()

    pre = prefilter_filings(tickers=args.tickers,
                            refresh=args.refresh_prefilter)
    if pre.empty:
        print("No 10-K filings found")
        return
    selected, gate_note = select_for_extraction(pre, budget=args.budget)
    print(f"prefilter: {int(pre['matched'].sum())}/{len(pre)} filings matched")
    print(gate_note)
    if args.prefilter_only:
        return

    if not args.build_only:
        counters = extract_all(selected, workers=args.workers)
        print(f"extraction: {counters}")

    df, stats = build_principal_customers(pre)
    validation_report(df, stats, pre)


if __name__ == "__main__":
    main()
