"""Factor 10 — 10-Q TF-IDF cosine vs the filing ~365 days earlier (same fiscal
quarter, controls seasonality; comparing adjacent quarters would not).

Usage:
    python -m alpha_graph.signals.lazy_prices_10q [--tickers AAPL MSFT]
"""

from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from alpha_graph.config import CACHE_DIR
from alpha_graph.signals.lazy_prices import _load_filing_texts

# Year-over-year matching window (days). 365 ± ~50d covers normal filing jitter
# and the occasional skipped quarter without matching the wrong fiscal period.
MIN_GAP = 300
MAX_GAP = 430
TARGET_GAP = 365

# Same vectorizer as factor 1 (cosine_similarity) for comparability.
_VEC_KWARGS = dict(max_features=10_000, stop_words="english", ngram_range=(1, 2), min_df=1)


def _match_yoy(dates: list[pd.Timestamp], i: int) -> int | None:
    """Index of the prior 10-Q closest to ~365 days before filing i, or None."""
    best, best_err = None, None
    for j in range(i):
        gap = (dates[i] - dates[j]).days
        if MIN_GAP <= gap <= MAX_GAP:
            err = abs(gap - TARGET_GAP)
            if best_err is None or err < best_err:
                best, best_err = j, err
    return best


def compute_for_ticker(ticker: str) -> list[dict]:
    filings = _load_filing_texts(ticker, "10-Q")
    if len(filings) < 4:
        logger.debug(f"[{ticker}] need >=4 10-Qs for YoY, have {len(filings)}")
        return []
    dates = [pd.Timestamp(f["filing_date"]) for f in filings]

    rows = []
    for i in range(len(filings)):
        j = _match_yoy(dates, i)
        if j is None:
            continue
        try:
            tfidf = TfidfVectorizer(**_VEC_KWARGS).fit_transform(
                [filings[j]["text"], filings[i]["text"]]
            )
            sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except ValueError:
            logger.warning(f"[{ticker}] TF-IDF failed at {filings[i]['filing_date']}")
            continue
        rows.append({
            "ticker": ticker,
            "filing_date": filings[i]["filing_date"],
            "prev_filing_date": filings[j]["filing_date"],
            "gap_days": (dates[i] - dates[j]).days,
            "cos_10q_yoy": round(sim, 6),
        })
    return rows


def compute_all(tickers: list[str] | None = None) -> pd.DataFrame:
    from alpha_graph.config import FILINGS_DIR
    if tickers is None:
        tickers = sorted(
            d.name for d in FILINGS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
    logger.info(f"Factor 10 (10-Q YoY cosine) for {len(tickers)} tickers...")
    out = []
    for t in tickers:
        out.extend(compute_for_ticker(t))

    df = pd.DataFrame(out)
    if df.empty:
        logger.warning("No 10-Q YoY pairs computed — is the 10-Q corpus present?")
        return df
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    path = CACHE_DIR / "lazy_prices_10Q_yoy.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(
        f"Saved {len(df)} pairs ({df['ticker'].nunique()} tickers) to {path}. "
        f"cos mean={df['cos_10q_yoy'].mean():.4f} std={df['cos_10q_yoy'].std():.4f} "
        f"median gap={df['gap_days'].median():.0f}d"
    )
    return df


def main():
    ap = argparse.ArgumentParser(description="Factor 10: 10-Q year-over-year Lazy Prices cosine")
    ap.add_argument("--tickers", nargs="+")
    args = ap.parse_args()
    df = compute_all(tickers=args.tickers)
    if not df.empty:
        print(df.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
