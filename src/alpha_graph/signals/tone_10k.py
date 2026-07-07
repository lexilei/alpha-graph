"""Factor 12 — change in Loughran-McDonald negative-word density vs the prior
10-K. Positive = filing got more negative. Static lexicon: PIT-safe.

Usage:
    python -m alpha_graph.signals.tone_10k [--tickers AAPL MSFT]
"""

from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR
from alpha_graph.signals.lazy_prices import _load_filing_texts

_LM = None


def _lm():
    global _LM
    if _LM is None:
        import pysentiment2 as ps
        _LM = ps.LM()
    return _LM


def _neg_prop(text: str) -> float | None:
    """Loughran-McDonald negative-word proportion of a document."""
    lm = _lm()
    toks = lm.tokenize(text)
    if len(toks) < 100:          # too short to be a real filing body
        return None
    score = lm.get_score(toks)
    return score["Negative"] / len(toks)


def compute_for_ticker(ticker: str) -> list[dict]:
    filings = _load_filing_texts(ticker, "10-K")
    if len(filings) < 2:
        return []
    rows = []
    prev_np = None
    prev_date = None
    for f in filings:
        np_ = _neg_prop(f["text"])
        if np_ is None:
            continue
        if prev_np is not None:
            rows.append({
                "ticker": ticker,
                "filing_date": f["filing_date"],
                "prev_filing_date": prev_date,
                "neg_prop": round(np_, 6),
                "tone_shift_10k": round(np_ - prev_np, 6),
            })
        prev_np, prev_date = np_, f["filing_date"]
    return rows


def compute_all(tickers: list[str] | None = None) -> pd.DataFrame:
    from alpha_graph.config import FILINGS_DIR
    if tickers is None:
        tickers = sorted(
            d.name for d in FILINGS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
    logger.info(f"Factor 12 (10-K LM tone shift) for {len(tickers)} tickers...")
    out = []
    for t in tickers:
        out.extend(compute_for_ticker(t))

    df = pd.DataFrame(out)
    if df.empty:
        logger.warning("No tone pairs computed — is the 10-K corpus present?")
        return df
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    path = CACHE_DIR / "tone_10k.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(
        f"Saved {len(df)} pairs ({df['ticker'].nunique()} tickers) to {path}. "
        f"neg_prop mean={df['neg_prop'].mean():.4f}, "
        f"tone_shift std={df['tone_shift_10k'].std():.4f}"
    )
    return df


def main():
    ap = argparse.ArgumentParser(description="Factor 12: 10-K Loughran-McDonald tone shift")
    ap.add_argument("--tickers", nargs="+")
    args = ap.parse_args()
    df = compute_all(tickers=args.tickers)
    if not df.empty:
        print(df.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
