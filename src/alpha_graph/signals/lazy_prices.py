"""Factor 1 — TF-IDF cosine between consecutive same-type filings
(Cohen-Malloy-Nguyen 2020 "Lazy Prices").

Usage:
    python -m alpha_graph.signals.lazy_prices [--tickers AAPL MSFT] [--form 10-K]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from alpha_graph.config import CACHE_DIR, FILINGS_DIR


def _load_filing_texts(ticker: str, form_type: str) -> list[dict]:
    """Load filing texts for a ticker, sorted chronologically.

    Returns list of dicts with keys: filing_date, text, accession_number.
    """
    ticker_dir = FILINGS_DIR / ticker
    if not ticker_dir.exists():
        return []

    filings = []
    for f in sorted(ticker_dir.glob(f"{form_type}_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        sections = data.get("sections", {})
        # Concatenate all extracted sections into one text block
        combined = "\n\n".join(sections.values())
        if len(combined) < 100:
            logger.debug(f"[{ticker}] Skipping thin filing {f.name}")
            continue
        filings.append({
            "filing_date": data["filing_date"],
            "text": combined,
            "accession_number": data.get("accession_number", ""),
        })

    return filings


def compute_similarity_for_ticker(ticker: str, form_type: str = "10-K") -> list[dict]:
    """Compute cosine similarity between consecutive filings for one ticker.

    Returns list of dicts with: ticker, filing_date, prev_filing_date,
    cosine_similarity, text_length_change.
    """
    filings = _load_filing_texts(ticker, form_type)

    if len(filings) < 2:
        logger.debug(f"[{ticker}] Need >= 2 {form_type} filings, have {len(filings)}")
        return []

    results = []

    for i in range(1, len(filings)):
        prev_text = filings[i - 1]["text"]
        curr_text = filings[i]["text"]

        # TF-IDF cosine similarity between the two documents
        vectorizer = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        try:
            tfidf_matrix = vectorizer.fit_transform([prev_text, curr_text])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except ValueError:
            logger.warning(f"[{ticker}] TF-IDF failed for {filings[i]['filing_date']}")
            continue

        results.append({
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": filings[i]["filing_date"],
            "prev_filing_date": filings[i - 1]["filing_date"],
            "cosine_similarity": round(float(sim), 6),
            "text_length_curr": len(curr_text),
            "text_length_prev": len(prev_text),
            "text_length_change_pct": round(
                (len(curr_text) - len(prev_text)) / max(len(prev_text), 1) * 100, 2
            ),
        })

    return results


def compute_all(
    tickers: list[str] | None = None,
    form_type: str = "10-K",
) -> pd.DataFrame:
    """Compute Lazy Prices similarity scores for all tickers with downloaded filings.

    Returns a DataFrame with one row per filing-pair comparison.
    """
    if tickers is None:
        # Discover tickers from downloaded filings
        if not FILINGS_DIR.exists():
            logger.error(f"No filings directory at {FILINGS_DIR}. Run filings download first.")
            return pd.DataFrame()
        tickers = sorted(
            d.name for d in FILINGS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    logger.info(f"Computing Lazy Prices similarity for {len(tickers)} tickers ({form_type})...")
    all_results = []

    for ticker in tickers:
        results = compute_similarity_for_ticker(ticker, form_type)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    if df.empty:
        logger.warning("No similarity scores computed — are filings downloaded?")
        return df

    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    # Save to cache
    out_path = CACHE_DIR / f"lazy_prices_{form_type.replace('-', '')}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} scores to {out_path}")

    # Summary stats
    logger.info(
        f"Similarity stats: mean={df['cosine_similarity'].mean():.4f}, "
        f"std={df['cosine_similarity'].std():.4f}, "
        f"min={df['cosine_similarity'].min():.4f}, "
        f"max={df['cosine_similarity'].max():.4f}"
    )

    return df


def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw similarity scores into a trading signal.

    Signal logic (per filing date cross-section):
    - Low similarity (bottom quintile) -> SHORT signal (-1)
    - High similarity (top quintile) -> LONG signal (+1)
    - Middle -> NEUTRAL (0)

    This follows the original Lazy Prices paper: firms that change their
    filings substantially tend to underperform.
    """
    if df.empty:
        return df

    df = df.copy()

    def _quintile_signal(group: pd.DataFrame) -> pd.Series:
        q20 = group["cosine_similarity"].quantile(0.2)
        q80 = group["cosine_similarity"].quantile(0.8)
        signal = pd.Series(0, index=group.index)
        signal[group["cosine_similarity"] <= q20] = -1  # big changes -> short
        signal[group["cosine_similarity"] >= q80] = 1   # small changes -> long
        return signal

    # Group by filing date (approximate — group by year for annual filings)
    df["filing_year"] = df["filing_date"].dt.year
    signals = []
    for _, group in df.groupby("filing_year"):
        signals.append(_quintile_signal(group))
    df["signal"] = pd.concat(signals)

    return df


def main():
    parser = argparse.ArgumentParser(description="Compute Lazy Prices similarity scores")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--form", default="10-K", choices=["10-K", "10-Q"])
    args = parser.parse_args()

    df = compute_all(tickers=args.tickers, form_type=args.form)
    if not df.empty:
        df = generate_signal(df)
        out_path = CACHE_DIR / f"lazy_prices_signal_{args.form.replace('-', '')}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Signal saved to {out_path}")

        # Print sample
        print("\n--- Sample Lazy Prices Scores ---")
        print(df[["ticker", "filing_date", "cosine_similarity", "signal"]].tail(20).to_string())


if __name__ == "__main__":
    main()
