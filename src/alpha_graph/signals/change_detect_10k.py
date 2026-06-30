"""Factor 14 — 10-K new-content fraction (chunk-alignment change detection).

The embedding-similarity factors (11/13) saturate near 1.0 because pooling a
whole 10-K into one vector is dominated by "which company is this", not the
year-over-year edits. This factor measures the edits directly: it aligns the
NEW filing's paragraphs against the PRIOR filing's paragraphs and counts how
much of the new filing has no good match in the old one — i.e. genuinely added
content. This is the embedding-space version of the Lazy Prices "added text"
idea, and it targets the exact failure mode we observed (a single added risk-
factor paragraph is 1/80 of the doc and gets averaged away by pooling; here it
is surfaced as an unmatched chunk).

Per consecutive 10-K pair (old, new):
  - split each into ~paragraph-sized chunks, embed + L2-normalize;
  - for each NEW chunk, best cosine match against any OLD chunk;
  - `new_content_frac` = share of new chunks whose best match < MATCH_THRESH
    (factor 14, threshold form);
  - `mean_novelty`    = 1 - mean(best match)   (threshold-free companion col).

MATCH_THRESH is a hyperparameter — pre-register it; defaults are a starting
point, not a tuned value.

Usage:
    python -m alpha_graph.signals.change_detect_10k [--tickers AAPL MSFT] \
        [--model BAAI/bge-base-en-v1.5] [--device mps] [--thresh 0.9]
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR
from alpha_graph.signals.lazy_prices import _load_filing_texts

TARGET_WORDS = 100        # paragraph-ish chunk size for change resolution
MIN_WORDS = 15
MAX_CHUNKS = 150
MATCH_THRESH = 0.90       # below this best-match cosine => "new" content


def _chunks(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, cur, curw = [], [], 0
    for p in paras:
        w = p.split()
        cur.extend(w)
        curw += len(w)
        if curw >= TARGET_WORDS:
            chunks.append(" ".join(cur))
            cur, curw = [], 0
    if curw >= MIN_WORDS:
        chunks.append(" ".join(cur))
    return chunks[:MAX_CHUNKS]


def compute_for_ticker(model, ticker: str, thresh: float) -> list[dict]:
    filings = _load_filing_texts(ticker, "10-K")
    if len(filings) < 2:
        return []
    rows = []
    prev_chunks = None
    prev_date = None
    for f in filings:
        ch = _chunks(f["text"])
        if len(ch) < 3:
            prev_chunks, prev_date = None, None
            continue
        if prev_chunks is not None:
            emb_new = model.encode(ch, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            emb_old = model.encode(prev_chunks, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            best = (emb_new @ emb_old.T).max(axis=1)
            rows.append({
                "ticker": ticker,
                "filing_date": f["filing_date"],
                "prev_filing_date": prev_date,
                "n_chunks_new": len(ch),
                "new_content_frac": round(float((best < thresh).mean()), 6),
                "mean_novelty": round(float(1 - best.mean()), 6),
            })
        prev_chunks, prev_date = ch, f["filing_date"]
    return rows


def compute_all(model_name: str, device: str, thresh: float, tickers: list[str] | None = None) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    from alpha_graph.config import FILINGS_DIR

    logger.info(f"Loading {model_name} on {device} (factor 14, thresh={thresh})...")
    model = SentenceTransformer(model_name, device=device)

    if tickers is None:
        tickers = sorted(
            d.name for d in FILINGS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
    logger.info(f"Change-detection for {len(tickers)} tickers...")
    out = []
    for i, t in enumerate(tickers, 1):
        out.extend(compute_for_ticker(model, t, thresh))
        if i % 25 == 0:
            logger.info(f"  {i}/{len(tickers)} tickers")

    df = pd.DataFrame(out)
    if df.empty:
        logger.warning("No pairs computed — is the 10-K corpus present?")
        return df
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["model"], df["thresh"] = model_name, thresh
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    path = CACHE_DIR / "change_detect_10k.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} pairs ({df['ticker'].nunique()} tickers) to {path}. "
                f"new_content_frac mean={df['new_content_frac'].mean():.4f}, "
                f"mean_novelty mean={df['mean_novelty'].mean():.4f}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Factor 14: 10-K new-content fraction")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--thresh", type=float, default=MATCH_THRESH)
    ap.add_argument("--tickers", nargs="+")
    args = ap.parse_args()
    df = compute_all(args.model, args.device, args.thresh, tickers=args.tickers)
    if not df.empty:
        print(df.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
