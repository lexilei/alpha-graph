"""LLM-enhanced filing change detection.

Goes beyond Lazy Prices cosine similarity to extract *what specifically changed*
between consecutive filings using GPT-4o-mini. Focuses on:
- New/removed risk factors
- Litigation language changes
- Revenue/guidance language shifts
- Material changes in MD&A tone

Each change is scored for materiality (-2 to +2) to generate a directional signal.

Usage:
    python -m alpha_graph.signals.filing_changes [--tickers AAPL MSFT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from alpha_graph.utils.llm import extract_json

from alpha_graph.config import CACHE_DIR, FILINGS_DIR, cfg

CHANGE_ANALYSIS_PROMPT = """\
You are a financial analyst comparing two consecutive SEC filings from the same company.

## Previous Filing ({prev_date})
{prev_text}

## Current Filing ({curr_date})
{curr_text}

Analyze the material changes between these filings. For each change, provide:
1. **category**: one of [risk_factor, litigation, revenue_guidance, operational, regulatory, management, debt_financing, other]
2. **description**: 1-2 sentence summary of what changed
3. **materiality_score**: integer from -2 (very bearish) to +2 (very bullish), 0 = neutral
4. **evidence**: key phrase or sentence from the filing supporting this change

Respond with a JSON object:
{{
    "changes": [
        {{
            "category": "...",
            "description": "...",
            "materiality_score": 0,
            "evidence": "..."
        }}
    ],
    "overall_score": 0,
    "summary": "1-2 sentence overall assessment"
}}

Focus on MATERIAL changes only — ignore boilerplate updates, date changes, and minor wording tweaks.
If there are no material changes, return an empty changes list with overall_score 0.
"""

# Truncate filing sections to fit in context window
MAX_SECTION_CHARS = 15_000


def _truncate(text: str, max_chars: int = MAX_SECTION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]"


def _load_filing_pairs(ticker: str, form_type: str = "10-K") -> list[tuple[dict, dict]]:
    """Load consecutive filing pairs for a ticker."""
    ticker_dir = FILINGS_DIR / ticker
    if not ticker_dir.exists():
        return []

    filings = []
    for f in sorted(ticker_dir.glob(f"{form_type}_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        sections = data.get("sections", {})
        # Prioritize Risk Factors and MD&A for change analysis
        text = ""
        for key in ["Item 1A - Risk Factors", "Risk Factors", "Item 7 - MD&A", "MD&A"]:
            if key in sections:
                text += f"\n\n### {key}\n{sections[key]}"
        if not text:
            text = "\n\n".join(sections.values())
        if len(text) < 100:
            continue
        data["_combined_text"] = text
        filings.append(data)

    pairs = []
    for i in range(1, len(filings)):
        pairs.append((filings[i - 1], filings[i]))
    return pairs


def analyze_filing_changes(
    prev_filing: dict,
    curr_filing: dict,
) -> dict:
    """Use LLM to analyze changes between two filings.

    Returns the parsed JSON response with changes, scores, and summary.
    """
    from openai import OpenAI

    if not cfg.llm_api_key:
        raise ValueError("LLM_API_KEY not set. Add it to .env")

    client = OpenAI(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url or None,
    )

    prompt = CHANGE_ANALYSIS_PROMPT.format(
        prev_date=prev_filing["filing_date"],
        curr_date=curr_filing["filing_date"],
        prev_text=_truncate(prev_filing["_combined_text"]),
        curr_text=_truncate(curr_filing["_combined_text"]),
    )

    response = client.chat.completions.create(
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    try:
        result = extract_json(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to parse LLM response: {e}")
        result = {"changes": [], "overall_score": 0, "summary": "Parse error"}

    result["ticker"] = curr_filing.get("ticker", "")
    result["filing_date"] = curr_filing["filing_date"]
    result["prev_filing_date"] = prev_filing["filing_date"]
    result["form_type"] = curr_filing.get("form_type", "")

    return result


def analyze_ticker(
    ticker: str,
    form_type: str = "10-K",
    cache: bool = True,
) -> list[dict]:
    """Analyze all consecutive filing pairs for a ticker."""
    cache_dir = CACHE_DIR / "filing_changes" / ticker
    cache_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_filing_pairs(ticker, form_type)
    if not pairs:
        logger.debug(f"[{ticker}] No filing pairs found for {form_type}")
        return []

    results = []
    for prev_filing, curr_filing in pairs:
        cache_key = f"{form_type}_{curr_filing['filing_date']}.json"
        cache_path = cache_dir / cache_key

        if cache and cache_path.exists():
            results.append(json.loads(cache_path.read_text(encoding="utf-8")))
            continue

        logger.info(
            f"[{ticker}] Analyzing {form_type} changes: "
            f"{prev_filing['filing_date']} -> {curr_filing['filing_date']}"
        )
        result = analyze_filing_changes(prev_filing, curr_filing)
        results.append(result)

        if cache:
            cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert analysis results into a flat DataFrame for signal generation."""
    rows = []
    for r in results:
        n_changes = len(r.get("changes", []))
        category_scores = {}
        for change in r.get("changes", []):
            cat = change.get("category", "other")
            score = change.get("materiality_score", 0)
            category_scores.setdefault(cat, []).append(score)

        row = {
            "ticker": r.get("ticker", ""),
            "filing_date": r.get("filing_date", ""),
            "prev_filing_date": r.get("prev_filing_date", ""),
            "form_type": r.get("form_type", ""),
            "overall_score": r.get("overall_score", 0),
            "n_changes": n_changes,
            "summary": r.get("summary", ""),
        }

        # Add per-category average scores
        for cat in ["risk_factor", "litigation", "revenue_guidance", "operational",
                     "regulatory", "management", "debt_financing"]:
            scores = category_scores.get(cat, [])
            row[f"score_{cat}"] = sum(scores) / len(scores) if scores else 0.0

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def analyze_all(
    tickers: list[str] | None = None,
    form_type: str = "10-K",
) -> pd.DataFrame:
    """Run LLM change analysis for all tickers with downloaded filings."""
    if tickers is None:
        if not FILINGS_DIR.exists():
            logger.error("No filings directory. Run filings download first.")
            return pd.DataFrame()
        tickers = sorted(
            d.name for d in FILINGS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    logger.info(f"Analyzing filing changes for {len(tickers)} tickers ({form_type})...")
    all_results = []

    for ticker in tickers:
        results = analyze_ticker(ticker, form_type)
        all_results.extend(results)

    df = results_to_dataframe(all_results)
    if not df.empty:
        out_path = CACHE_DIR / f"filing_changes_{form_type.replace('-', '')}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df)} change analyses to {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="LLM-enhanced filing change detection")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--form", default="10-K", choices=["10-K", "10-Q"])
    args = parser.parse_args()

    df = analyze_all(tickers=args.tickers, form_type=args.form)
    if not df.empty:
        print("\n--- Filing Change Analysis ---")
        print(df[["ticker", "filing_date", "overall_score", "n_changes", "summary"]].to_string())


if __name__ == "__main__":
    main()
