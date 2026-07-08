"""Factor 14 — fraction of the new 10-K's paragraph chunks with no good
embedding match in the prior filing (genuinely added content). Avoids the
whole-doc similarity saturation that defeats factors 11/13.

Usage:
    python -m alpha_graph.signals.change_detect_10k [--tickers AAPL MSFT] \
        [--model BAAI/bge-base-en-v1.5] [--device mps] [--thresh 0.9]
"""

from __future__ import annotations

import argparse

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
    # fixed word windows: bounded chunk size (paragraph accumulation produced
    # unbounded "monster chunks" on newline-poor full_text filings)
    w = text.split()
    out = [" ".join(w[i:i + TARGET_WORDS]) for i in range(0, len(w), TARGET_WORDS)]
    if out and len(out[-1].split()) < MIN_WORDS:
        out.pop()
    return out[:MAX_CHUNKS]


def compute_for_ticker(model, ticker: str, thresh: float) -> list[dict]:
    filings = _load_filing_texts(ticker, "10-K")
    if len(filings) < 2:
        return []
    rows = []
    prev_emb = None
    prev_date = None
    for f in filings:
        ch = _chunks(f["text"])
        if len(ch) < 3:
            prev_emb, prev_date = None, None
            continue
        # embed each filing once; reuse as the "old" side of the next pair
        emb = model.encode(ch, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        if prev_emb is not None:
            best = (emb @ prev_emb.T).max(axis=1)
            rows.append({
                "ticker": ticker,
                "filing_date": f["filing_date"],
                "prev_filing_date": prev_date,
                "n_chunks_new": len(ch),
                "new_content_frac": round(float((best < thresh).mean()), 6),
                "mean_novelty": round(float(1 - best.mean()), 6),
            })
        prev_emb, prev_date = emb, f["filing_date"]
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
    # Per-ticker checkpoint: resume skips tickers with a part (or .empty) file.
    parts_dir = CACHE_DIR / "change_detect_10k_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Change-detection for {len(tickers)} tickers...")
    out, n_cached = [], 0
    for i, t in enumerate(tickers, 1):
        part = parts_dir / f"{t}.parquet"
        if part.exists():
            out.append(pd.read_parquet(part))
            n_cached += 1
        elif not (parts_dir / f"{t}.empty").exists():
            rows = compute_for_ticker(model, t, thresh)
            if rows:
                pdf = pd.DataFrame(rows)
                tmp = part.with_suffix(".parquet.tmp")
                pdf.to_parquet(tmp, index=False)
                tmp.replace(part)
                out.append(pdf)
            else:
                (parts_dir / f"{t}.empty").touch()
        if i % 25 == 0:
            logger.info(f"  {i}/{len(tickers)} tickers ({n_cached} from cache)")

    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
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
