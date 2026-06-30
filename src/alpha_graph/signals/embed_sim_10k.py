"""Factors 11 / 13 — 10-K semantic similarity via sentence-transformer embeddings.

The semantic counterpart to factor 1 (TF-IDF cosine). Same question — "how
much did the 10-K change?" — but in embedding space, so paraphrase and word
reordering don't count as change. One module, swappable encoder, so factors 1
(bag-of-words), 11 (finance-tuned), and 13 (general bge) form a clean 3-way A/B
on the SAME filing pairs; `factor_orthogonality.py` then says which encoder, if
any, carries incremental IC the others miss.

10-K bodies far exceed the 512-token model limit, so each filing is split into
~350-word chunks, every chunk embedded and L2-normalized, then mean-pooled into
one document vector. Similarity = cosine between consecutive filings' document
vectors (same consecutive pairing as factor 1).

Determinism: CPU inference with a pinned model is reproducible. `--device mps`
is faster on Apple Silicon at the cost of tiny float nondeterminism (immaterial
to the signal). Model name/revision is recorded in the output for provenance.

Usage:
    python -m alpha_graph.signals.embed_sim_10k --tag fin \
        --model FinLang/finance-embeddings-investopedia            # factor 11
    python -m alpha_graph.signals.embed_sim_10k --tag bge \
        --model BAAI/bge-base-en-v1.5                              # factor 13
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR
from alpha_graph.signals.lazy_prices import _load_filing_texts

WORDS_PER_CHUNK = 350     # ~< 512 tokens
MAX_CHUNKS = 80           # cap worst-case cost per filing


def _chunks(text: str) -> list[str]:
    w = text.split()
    out = [" ".join(w[i:i + WORDS_PER_CHUNK]) for i in range(0, len(w), WORDS_PER_CHUNK)]
    return out[:MAX_CHUNKS]


def _doc_vec(model, text: str) -> np.ndarray | None:
    ch = _chunks(text)
    if not ch:
        return None
    emb = model.encode(ch, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    v = emb.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def compute_for_ticker(model, ticker: str, col: str) -> list[dict]:
    filings = _load_filing_texts(ticker, "10-K")
    if len(filings) < 2:
        return []
    rows = []
    prev_vec = None
    prev_date = None
    for f in filings:
        vec = _doc_vec(model, f["text"])
        if vec is None:
            continue
        if prev_vec is not None:
            sim = float(np.dot(prev_vec, vec))
            rows.append({
                "ticker": ticker,
                "filing_date": f["filing_date"],
                "prev_filing_date": prev_date,
                col: round(sim, 6),
            })
        prev_vec, prev_date = vec, f["filing_date"]
    return rows


def compute_all(model_name: str, tag: str, device: str, tickers: list[str] | None = None) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    from alpha_graph.config import FILINGS_DIR

    col = f"embed_sim_10k_{tag}"
    logger.info(f"Loading {model_name} on {device} for factor column '{col}'...")
    model = SentenceTransformer(model_name, device=device)

    if tickers is None:
        tickers = sorted(
            d.name for d in FILINGS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
    logger.info(f"Embedding-similarity for {len(tickers)} tickers...")
    out = []
    for i, t in enumerate(tickers, 1):
        out.extend(compute_for_ticker(model, t, col))
        if i % 25 == 0:
            logger.info(f"  {i}/{len(tickers)} tickers")

    df = pd.DataFrame(out)
    if df.empty:
        logger.warning("No pairs computed — is the 10-K corpus present?")
        return df
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["model"] = model_name
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    path = CACHE_DIR / f"embed_sim_10k_{tag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} pairs ({df['ticker'].nunique()} tickers) to {path}. "
                f"{col} mean={df[col].mean():.4f} std={df[col].std():.4f}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Factors 11/13: 10-K embedding similarity")
    ap.add_argument("--model", required=True, help="sentence-transformers model name")
    ap.add_argument("--tag", required=True, help="column/file suffix, e.g. 'fin' or 'bge'")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--tickers", nargs="+")
    args = ap.parse_args()
    df = compute_all(args.model, args.tag, args.device, tickers=args.tickers)
    if not df.empty:
        print(df.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
