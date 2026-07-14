"""Extract inter-company relationships from 10-K filings via LLM.

Reads the Business Description section of each 10-K filing and uses
DeepSeek-V3 to extract supplier, customer, competitor, and partner
relationships. Only relationships where BOTH companies are in our
universe are kept.

The resulting directed graph enables cross-firm signal propagation:
when Company A files bad news, predict the impact on its suppliers
and customers before the market prices it in.

Academic basis: Cohen & Frazzini (2008) "Economic Links and Predictable
Returns" — supplier stock returns lag customer stock returns by one month.

Usage:
    python -m alpha_graph.data.relationships [--tickers AAPL MSFT]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR, cfg
from alpha_graph.utils.llm import extract_json

RELATIONSHIP_TYPES = ["supplier", "customer", "competitor", "partner"]

# Truncate Business Description to fit context window
MAX_SECTION_CHARS = 20_000

RELATIONSHIP_CACHE_DIR = CACHE_DIR / "relationships"

EXTRACTION_PROMPT = """\
You are a financial analyst extracting business relationships from an SEC 10-K filing.

## Company: {ticker} ({company_name})

## Filing Text (Business Description)
{business_text}

## Universe of Companies (ONLY report relationships with these tickers)
{ticker_list}

For each company in the universe that {ticker} has a DIRECT business relationship with,
classify the relationship FROM {ticker}'s perspective:
- supplier: {ticker} BUYS FROM this company (they supply goods/services to {ticker})
- customer: {ticker} SELLS TO this company (they buy {ticker}'s goods/services)
- competitor: {ticker} COMPETES with this company in the same market
- partner: {ticker} has a strategic partnership, joint venture, or alliance

Respond with JSON:
{{
    "relationships": [
        {{
            "target_ticker": "...",
            "relation": "supplier|customer|competitor|partner",
            "confidence": 0.5-1.0,
            "evidence": "brief phrase from filing supporting this"
        }}
    ]
}}

Rules:
- ONLY include relationships explicitly mentioned or strongly implied in the text
- Do NOT infer relationships from general industry knowledge
- confidence >= 0.8 for explicitly named, 0.5-0.8 for strongly implied
- If no relationships found with universe companies, return empty list
- Each target_ticker must be from the universe list above
"""


def _get_universe_tickers() -> list[str]:
    """Get list of tickers with downloaded filings."""
    if not FILINGS_DIR.exists():
        return []
    return sorted(
        d.name for d in FILINGS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def _load_business_text(filing_path: Path) -> tuple[str, str, str]:
    """Load Business Description from a 10-K filing.

    Returns (business_text, company_name, filing_date).
    Falls back to full section text if no Business section.
    """
    data = json.loads(filing_path.read_text(encoding="utf-8"))
    sections = data.get("sections", {})

    # Prefer Business Description
    text = ""
    for key in ["Item 1 - Business", "Business", "Item 1 - Business Description"]:
        if key in sections and len(sections[key]) > 100:
            text = sections[key]
            break

    # Fallback: concatenate all sections
    if not text:
        text = "\n\n".join(sections.values())

    if len(text) > MAX_SECTION_CHARS:
        text = text[:MAX_SECTION_CHARS] + "\n\n[... truncated ...]"

    return text, data.get("company_name", ""), data.get("filing_date", "")


def _parse_confidence(value) -> float | None:
    """Parse an LLM-reported confidence into a float clamped to [0, 1].

    Returns None for missing/None/NaN/inf/unparseable values so callers
    drop the row explicitly (and count it) instead of letting NaN flow
    through and silently fail every `>= min_conf` comparison.
    """
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(conf):
        return None
    return max(0.0, min(1.0, conf))


def _normalize_ws(text: str) -> str:
    """Lowercase and collapse whitespace runs to single spaces."""
    return " ".join(text.split()).lower()


def _evidence_in_text(evidence, text: str) -> bool:
    """Whitespace-normalized, case-insensitive substring check.

    True iff the evidence quote (whitespace collapsed, lowercased)
    appears in the section text (same normalization). Empty or
    non-string evidence never verifies.
    """
    if not isinstance(evidence, str) or not evidence.strip() or not text:
        return False
    return _normalize_ws(evidence) in _normalize_ws(text)


def _validate_relationships(
    raw_rels: list[dict],
    universe_set: set[str],
    business_text: str,
) -> tuple[list[dict], int]:
    """Normalize raw LLM relationship dicts into validated rows.

    Keeps only rows whose target is in the universe and whose relation is
    a known type. Rows with missing/None/NaN confidence are dropped and
    counted (returned as the second element). Each kept row gains
    `evidence_verified`: whether the evidence quote appears verbatim
    (whitespace-normalized, case-insensitive) in the section text the
    LLM was shown. Failing rows are flagged, never dropped.
    """
    valid_rels: list[dict] = []
    n_bad_conf = 0
    for rel in raw_rels:
        target = rel.get("target_ticker", "")
        relation = rel.get("relation", "")
        if target not in universe_set or relation not in RELATIONSHIP_TYPES:
            continue
        conf = _parse_confidence(rel.get("confidence"))
        if conf is None:
            n_bad_conf += 1
            continue
        evidence = rel.get("evidence", "") or ""
        valid_rels.append({
            "target_ticker": target,
            "relation": relation,
            "confidence": conf,
            "evidence": evidence,
            "evidence_verified": _evidence_in_text(evidence, business_text),
        })
    return valid_rels, n_bad_conf


def extract_relationships_for_filing(
    ticker: str,
    filing_path: Path,
    universe_tickers: list[str],
    cache: bool = True,
) -> dict:
    """Extract relationships from a single 10-K filing via LLM.

    Returns dict with: ticker, filing_date, relationships list.
    Uses per-filing JSON cache to avoid redundant LLM calls.
    """
    from openai import OpenAI

    # Check cache
    filing_name = filing_path.stem  # e.g. "10-K_2024-11-01_000032019324000123"
    cache_dir = RELATIONSHIP_CACHE_DIR / ticker
    cache_path = cache_dir / f"{filing_name}.json"

    if cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    # Load filing text
    business_text, company_name, filing_date = _load_business_text(filing_path)
    if len(business_text) < 200:
        logger.debug(f"[{ticker}] Filing too short, skipping: {filing_path.name}")
        result = {"ticker": ticker, "filing_date": filing_date, "relationships": []}
        if cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    # Filter universe to exclude self
    other_tickers = [t for t in universe_tickers if t != ticker]
    ticker_list = ", ".join(other_tickers)

    if not cfg.llm_api_key:
        raise ValueError("LLM_API_KEY not set. Add it to .env")

    client = OpenAI(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url or None,
    )

    prompt = EXTRACTION_PROMPT.format(
        ticker=ticker,
        company_name=company_name,
        business_text=business_text,
        ticker_list=ticker_list,
    )

    response = client.chat.completions.create(
        model=cfg.llm_model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    try:
        parsed = extract_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        logger.error(f"[{ticker}] Failed to parse LLM response: {e}")
        parsed = {"relationships": []}

    # Validate: universe/type filter, explicit confidence guard, and the
    # evidence-substring check against the exact text the LLM was shown
    # (business_text is only in memory here, at extraction time).
    valid_rels, n_bad_conf = _validate_relationships(
        parsed.get("relationships", []), set(universe_tickers), business_text
    )
    if n_bad_conf:
        logger.debug(
            f"[{ticker}] {filing_date}: dropped {n_bad_conf} relationship(s) "
            f"with missing/None/NaN confidence"
        )

    result = {
        "ticker": ticker,
        "company_name": company_name,
        "filing_date": filing_date,
        "model": cfg.llm_model,
        "relationships": valid_rels,
    }

    # Cache
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    logger.info(
        f"[{ticker}] {filing_date}: {len(valid_rels)} relationships extracted"
    )
    return result


def extract_all_relationships(
    tickers: list[str] | None = None,
    form_type: str = "10-K",
    workers: int = 8,
) -> pd.DataFrame:
    """Extract relationships from all filings for all tickers.

    Runs `workers` concurrent LLM calls; each filing's result is cached to
    its own JSON, so an interrupted run resumes where it stopped.

    Returns DataFrame with: source, target, relation, confidence, evidence,
    evidence_verified, filing_date, model. Saves to
    CACHE_DIR / "relationships.parquet".

    Guards: rows with missing/None/NaN confidence are dropped and counted
    (per-filing caches written before this guard may carry them).
    evidence_verified is computed at extraction time against the section
    text the LLM saw; cached results that predate the flag are treated as
    un-verified (False) — use audit_evidence() to recompute offline.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    universe = _get_universe_tickers()
    if not universe:
        logger.error("No filings directory. Run filings download first.")
        return pd.DataFrame()

    if tickers is None:
        tickers = universe

    jobs: list[tuple[str, Path]] = []
    for ticker in tickers:
        ticker_dir = FILINGS_DIR / ticker
        if not ticker_dir.exists():
            continue
        jobs.extend((ticker, p) for p in sorted(ticker_dir.glob(f"{form_type}_*.json")))

    logger.info(
        f"Extracting relationships from {len(jobs)} {form_type} filings "
        f"({len(tickers)} tickers, model={cfg.llm_model}, {workers} workers)..."
    )

    all_rows = []
    n_done = n_failed = n_bad_conf = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(extract_relationships_for_filing, t, p, universe): (t, p)
            for t, p in jobs
        }
        for fut in as_completed(futures):
            ticker, fpath = futures[fut]
            n_done += 1
            try:
                result = fut.result()
            except Exception as e:
                n_failed += 1
                logger.error(f"[{ticker}] {fpath.name} failed: {e}")
                continue

            for rel in result.get("relationships", []):
                # Cached JSONs written before the confidence guard can
                # still carry missing/None/NaN — drop and count here too.
                conf = _parse_confidence(rel.get("confidence"))
                if conf is None:
                    n_bad_conf += 1
                    continue
                all_rows.append({
                    "source": ticker,
                    "target": rel["target_ticker"],
                    "relation": rel["relation"],
                    "confidence": conf,
                    "evidence": rel.get("evidence", ""),
                    # Caches from before the evidence check lack the flag:
                    # treat missing as un-verified.
                    "evidence_verified": bool(rel.get("evidence_verified", False)),
                    "filing_date": result["filing_date"],
                    "model": result.get("model", ""),
                })

            if n_done % 200 == 0:
                logger.info(
                    f"  Progress: {n_done}/{len(jobs)} filings "
                    f"({n_failed} failed), {len(all_rows)} edges so far"
                )

    if n_failed:
        logger.warning(f"{n_failed}/{len(jobs)} filings failed — rerun to retry (cached ones skip)")
    if n_bad_conf:
        logger.debug(f"Dropped {n_bad_conf} relationship row(s) with missing/None/NaN confidence")

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("No relationships extracted")
        return df

    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.sort_values(["source", "filing_date"]).reset_index(drop=True)

    # Save
    out_path = CACHE_DIR / "relationships.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} relationship edges to {out_path}")

    # Summary
    logger.info(f"  Unique source tickers: {df['source'].nunique()}")
    logger.info(f"  Unique target tickers: {df['target'].nunique()}")
    logger.info(f"  Relation distribution:")
    for rel, count in df["relation"].value_counts().items():
        logger.info(f"    {rel}: {count}")

    return df


def audit_evidence(
    relationships_df: pd.DataFrame,
    filings_dir: Path | str = FILINGS_DIR,
    max_filings: int | None = None,
) -> dict:
    """Offline audit: re-check each edge's `evidence` quote against filing text.

    Existing caches predate the extraction-time evidence check (their
    `evidence_verified` column is missing — treated as un-verified), so this
    recomputes verification from the filings on disk. For every
    (source, filing_date) group it loads the matching filing(s) under
    `filings_dir/<source>/` via the same loader extraction used (same
    section preference, same truncation) and tests whitespace-normalized,
    case-insensitive containment. A row verifies if its evidence appears
    in ANY same-day filing for that ticker.

    Read-only: modifies no cache. A full run loads every filing on disk
    (hours of IO at universe scale) — pass `max_filings` to smoke-test.

    Returns a report dict with: n_filings_requested, n_filings_found,
    n_filings_missing, n_rows_checked, n_verified, hit_rate, and `rows`
    (a DataFrame: source, target, relation, filing_date,
    evidence_verified_audit).
    """
    filings_dir = Path(filings_dir)
    df = relationships_df.copy()
    if "evidence_verified" not in df.columns:
        df["evidence_verified"] = False  # missing column = un-verified
    df["filing_date"] = pd.to_datetime(df["filing_date"])

    groups = list(df.groupby(["source", "filing_date"], sort=True))
    if max_filings is not None:
        groups = groups[:max_filings]

    rows = []
    n_filings_missing = 0
    for (source, fdate), g in groups:
        date_str = pd.Timestamp(fdate).strftime("%Y-%m-%d")
        candidates = sorted((filings_dir / source).glob(f"*_{date_str}_*.json"))
        texts = []
        for p in candidates:
            try:
                text, _, _ = _load_business_text(p)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                logger.debug(f"audit_evidence: cannot read {p}: {e}")
                continue
            texts.append(_normalize_ws(text))
        if not texts:
            n_filings_missing += 1
        for _, row in g.iterrows():
            evidence = row.get("evidence", "")
            verified = (
                isinstance(evidence, str)
                and bool(evidence.strip())
                and any(_normalize_ws(evidence) in t for t in texts)
            )
            rows.append({
                "source": source,
                "target": row["target"],
                "relation": row["relation"],
                "filing_date": fdate,
                "evidence_verified_audit": verified,
            })

    rows_df = pd.DataFrame(rows)
    n_checked = len(rows_df)
    n_verified = int(rows_df["evidence_verified_audit"].sum()) if n_checked else 0
    report = {
        "n_filings_requested": len(groups),
        "n_filings_found": len(groups) - n_filings_missing,
        "n_filings_missing": n_filings_missing,
        "n_rows_checked": n_checked,
        "n_verified": n_verified,
        "hit_rate": (n_verified / n_checked) if n_checked else float("nan"),
        "rows": rows_df,
    }
    logger.info(
        f"audit_evidence: {n_verified}/{n_checked} rows verified "
        f"(hit rate {report['hit_rate']:.1%}) over {len(groups)} filings, "
        f"{n_filings_missing} filing(s) not found on disk"
        if n_checked else "audit_evidence: no rows to check"
    )
    return report


def build_graph(
    relationships: pd.DataFrame | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> nx.DiGraph:
    """Build a directed graph from extracted relationships.

    Nodes = tickers. Edges = (source, target) with attributes:
    relation, confidence, filing_date.

    If as_of_date is provided, only includes relationships from
    filings filed on or before that date (prevents lookahead bias).

    For duplicate edges (same source-target pair), keeps the most
    recent filing's relationship.
    """
    if relationships is None:
        rel_path = CACHE_DIR / "relationships.parquet"
        if not rel_path.exists():
            logger.error("No relationships data. Run extraction first.")
            return nx.DiGraph()
        relationships = pd.read_parquet(rel_path)

    if relationships.empty:
        return nx.DiGraph()

    relationships = relationships.copy()
    relationships["filing_date"] = pd.to_datetime(relationships["filing_date"])

    # Filter by as_of_date
    if as_of_date is not None:
        as_of_date = pd.Timestamp(as_of_date)
        relationships = relationships[relationships["filing_date"] <= as_of_date]

    if relationships.empty:
        return nx.DiGraph()

    # Deduplicate: for same (source, target), keep most recent filing.
    # The sort must be a deterministic TOTAL order: the parquet contains
    # same-day duplicates (same pair, same filing_date, different
    # relation/confidence), and a single-key unstable sort made the
    # survivor depend on input row order. Policy: latest filing_date
    # wins; same-day ties broken by highest confidence; remaining ties
    # by lexicographically-last relation (arbitrary but fixed).
    relationships = (
        relationships.sort_values(
            ["filing_date", "confidence", "relation", "source", "target"],
            kind="stable",
        )
        .drop_duplicates(subset=["source", "target"], keep="last")
    )

    # Filter low-confidence edges
    relationships = relationships[relationships["confidence"] >= 0.5]

    # Build graph
    G = nx.DiGraph()

    # Add all universe tickers as nodes (even if isolated)
    universe = _get_universe_tickers()
    for t in universe:
        G.add_node(t)

    for _, row in relationships.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            relation=row["relation"],
            confidence=row["confidence"],
            filing_date=str(row["filing_date"].date()),
            evidence=row["evidence"],
        )

    return G


def graph_summary(G: nx.DiGraph) -> dict:
    """Print and return graph diagnostics."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_isolated = len(list(nx.isolates(G)))

    # Degree stats (undirected view)
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0

    # Edge type distribution
    edge_types = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relation", "unknown")
        edge_types[rel] = edge_types.get(rel, 0) + 1

    # Most connected nodes
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]

    summary = {
        "nodes": n_nodes,
        "edges": n_edges,
        "isolated_nodes": n_isolated,
        "connected_nodes": n_nodes - n_isolated,
        "avg_degree": round(avg_degree, 2),
        "edge_types": edge_types,
        "top_connected": [(t, d) for t, d in top_nodes],
    }

    logger.info(f"Graph: {n_nodes} nodes, {n_edges} edges, {n_isolated} isolated")
    logger.info(f"  Avg degree: {avg_degree:.1f}")
    logger.info(f"  Edge types: {edge_types}")
    logger.info(f"  Top connected: {top_nodes[:5]}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract inter-company relationships from 10-K filings")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--form", default="10-K", choices=["10-K", "10-Q"])
    parser.add_argument("--summary-only", action="store_true",
                        help="Only show graph summary (no extraction)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent LLM calls")
    args = parser.parse_args()

    if args.summary_only:
        G = build_graph()
        graph_summary(G)
        return

    df = extract_all_relationships(tickers=args.tickers, form_type=args.form, workers=args.workers)
    if df.empty:
        print("No relationships extracted")
        return

    print(f"\n--- Relationship Extraction Complete ---")
    print(f"Total edges: {len(df)}")
    print(f"\nSample edges:")
    print(df[["source", "target", "relation", "confidence"]].head(20).to_string())

    # Build and summarize graph
    G = build_graph(df)
    graph_summary(G)


if __name__ == "__main__":
    main()
