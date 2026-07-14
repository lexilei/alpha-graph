"""C21/C22 — paper-faithful Lazy Prices edit measures on consecutive same-type
10-Ks (Cohen-Malloy-Nguyen 2020, "Lazy Prices"). The one remaining text shot,
redirected to the paper's OWN document-distance measures instead of a cosine.

Two measures, both on the same pinned token stream:
  - C21 ``sim_minedit_10k`` — difflib's Ratcliff-Obershelp ``.ratio()`` (2M/T),
    the registered stand-in for the paper's token-level normalized edit
    distance.
  - C22 ``sim_simple_10k`` — CMN's Sim_Simple, ``1 - (adds+deletes)/(len_old+
    len_new)``, where adds/deletes are the inserted/deleted token counts from
    the SAME SequenceMatcher opcodes (a ``replace`` counting on both sides).

Both reuse C1's 10-K pair set verbatim (``lazy_prices._load_filing_texts`` —
same amendment/fallback contamination fixes — plus C1's >460-day consecutive-
pair gap filter) and stamp each score at the NEW filing's ``filing_date`` (same
availability as C1). The tokenizer is pinned: ``re.findall(r"[a-z0-9]+",
text.lower())`` — lowercase alphanumeric word tokens.

Runtime: ``.ratio()`` is the pinned path (benchmarked ~2s at 60k combined
tokens, ~16s at 200k). Pairs whose combined length exceeds
``FALLBACK_COMBINED_TOKENS`` (a handful of very large bank 10-Ks, up to ~2M
tokens, where the full ``.ratio()`` would exceed ~30s) use the documented
fallback INSIDE the same pinned family: ``.ratio()`` / Sim_Simple on 10k-token
chunks aligned sequentially and length-weighted. The chosen path is recorded
per row (``path`` column).

Note (mathematical): under these pinned implementations sim_simple reduces to
the ratio exactly — adds+deletes = len_old+len_new-2M, so
1-(adds+deletes)/(len_old+len_new) = 2M/(len_old+len_new) = ratio — for both
the full and the length-weighted chunked path. C21 and C22 therefore carry the
same value per pair; they are cached and evaluated as the two registered
measures regardless (the family A/B is decided on either).

Usage:
    python -m alpha_graph.signals.lazy_prices_edit [--tickers AAPL MSFT] \
        [--workers N]
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR
from alpha_graph.signals.lazy_prices import _load_filing_texts

# Pinned tokenizer: lowercase alphanumeric word tokens (no pure-punctuation
# tokens; digits kept). Compiled once, reused in every worker.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# C1's consecutive-pair filter: a >460-day gap is a dropped-year span (a filing
# was fallback-mode/thin and removed upstream), not a real consecutive pair.
MAX_GAP_DAYS = 460

# Above this combined (len_old + len_new) token count the full ``.ratio()``
# risks exceeding ~30s (calibrated: 200k -> ~16s, 250k -> ~34s), so the pair
# takes the chunked fallback instead. 98%+ of pairs stay on the pinned path.
FALLBACK_COMBINED_TOKENS = 200_000

# Chunk granularity for the fallback (documented "10k-token chunks").
CHUNK_TOKENS = 10_000

# Per-ticker checkpoint directory (resume + parallel-safe: each worker writes
# only its own file). The final cache is assembled from these parts.
PARTS_DIR = CACHE_DIR / "lazy_prices_edit_parts"
CACHE_FILE = CACHE_DIR / "lazy_prices_edit.parquet"


def tokenize(text: str) -> list[str]:
    """Pinned tokenizer: lowercase alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


def _adds_deletes(sm: difflib.SequenceMatcher) -> tuple[int, int]:
    """CMN Sim_Simple counts from SequenceMatcher opcodes: inserted (adds) and
    deleted (deletes) tokens, a ``replace`` contributing to BOTH sides."""
    adds = deletes = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            adds += j2 - j1
        elif tag == "delete":
            deletes += i2 - i1
        elif tag == "replace":
            deletes += i2 - i1
            adds += j2 - j1
    return adds, deletes


def _similarity_full(old: list[str], new: list[str]) -> tuple[float, float]:
    """Pinned path: one ``SequenceMatcher(autojunk=False)`` yields both
    measures. ``.ratio()`` and ``.get_opcodes()`` share the cached matching
    blocks, so the match is computed once."""
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    minedit = sm.ratio()
    adds, deletes = _adds_deletes(sm)
    denom = len(old) + len(new)
    simple = 1.0 - (adds + deletes) / denom if denom else 1.0
    return minedit, simple


def _chunks(tokens: list[str], size: int) -> list[list[str]]:
    return [tokens[i:i + size] for i in range(0, len(tokens), size)] or [[]]


def _similarity_chunked(old: list[str], new: list[str]) -> tuple[float, float]:
    """Documented fallback inside the pinned family for very large pairs:
    ``.ratio()`` / Sim_Simple on 10k-token chunks aligned sequentially (chunk
    k of old vs chunk k of new; a trailing chunk on the longer side aligns
    against an empty chunk), length-weighted by each aligned pair's combined
    token count."""
    total = len(old) + len(new)
    if total == 0:
        return 1.0, 1.0
    oc = _chunks(old, CHUNK_TOKENS)
    nc = _chunks(new, CHUNK_TOKENS)
    wsum_ratio = 0.0
    adds = deletes = 0
    for k in range(max(len(oc), len(nc))):
        a = oc[k] if k < len(oc) else []
        b = nc[k] if k < len(nc) else []
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        wsum_ratio += sm.ratio() * (len(a) + len(b))
        da, dd = _adds_deletes(sm)
        adds += da
        deletes += dd
    minedit = wsum_ratio / total
    simple = 1.0 - (adds + deletes) / total
    return minedit, simple


def pair_similarity(old: list[str], new: list[str]) -> tuple[float, float, str]:
    """Return ``(sim_minedit, sim_simple, path)`` picking the path by size."""
    if len(old) + len(new) > FALLBACK_COMBINED_TOKENS:
        m, s = _similarity_chunked(old, new)
        return m, s, "chunked"
    m, s = _similarity_full(old, new)
    return m, s, "full"


def compute_for_ticker(ticker: str) -> list[dict]:
    """All consecutive same-type 10-K pairs for one ticker (C1's pair set)."""
    filings = _load_filing_texts(ticker, "10-K")
    if len(filings) < 2:
        return []
    rows: list[dict] = []
    prev_tokens = tokenize(filings[0]["text"])
    prev_date = filings[0]["filing_date"]
    for f in filings[1:]:
        cur_tokens = tokenize(f["text"])
        gap = (pd.Timestamp(f["filing_date"]) - pd.Timestamp(prev_date)).days
        # prev always advances to the immediate predecessor (matches C1: a
        # skipped >460d span still compares the next filing to this one).
        if gap <= MAX_GAP_DAYS:
            minedit, simple, path = pair_similarity(prev_tokens, cur_tokens)
            rows.append({
                "ticker": ticker,
                "filing_date": f["filing_date"],
                "prev_filing_date": prev_date,
                "sim_minedit_10k": round(float(minedit), 6),
                "sim_simple_10k": round(float(simple), 6),
                "n_tokens_old": len(prev_tokens),
                "n_tokens_new": len(cur_tokens),
                "path": path,
            })
        prev_tokens, prev_date = cur_tokens, f["filing_date"]
    return rows


def _worker(ticker: str) -> tuple[str, str]:
    """Process-pool worker: compute one ticker, checkpoint its part file.
    Top-level and picklable so the spawn start method (macOS default) can
    re-import it — hence the ``__main__`` guard on the CLI below."""
    part = PARTS_DIR / f"{ticker}.parquet"
    empty = PARTS_DIR / f"{ticker}.empty"
    if part.exists():
        return ticker, "cached"
    if empty.exists():
        return ticker, "empty-cached"
    try:
        rows = compute_for_ticker(ticker)
    except Exception as e:  # never let one ticker abort the whole pool
        return ticker, f"ERROR: {e!r}"
    if rows:
        tmp = part.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        tmp.replace(part)
        return ticker, f"{len(rows)} pairs"
    empty.touch()
    return ticker, "empty"


def compute_all(
    tickers: list[str] | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    """Compute C21/C22 for all tickers with a 10-K corpus and cache the result.

    Parallelized per ticker with a process pool (SequenceMatcher is pure-Python
    CPU work, so processes beat threads). Returns the assembled DataFrame with
    one row per consecutive-10-K pair.
    """
    if tickers is None:
        if not FILINGS_DIR.exists():
            logger.error(f"No filings directory at {FILINGS_DIR}.")
            return pd.DataFrame()
        tickers = sorted(
            d.name for d in FILINGS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    logger.info(f"Computing C21/C22 for {len(tickers)} tickers on {workers} "
                f"workers (fallback > {FALLBACK_COMBINED_TOKENS:,} tokens)...")

    errors = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (tk, status) in enumerate(ex.map(_worker, tickers), 1):
            if status.startswith("ERROR"):
                errors.append((tk, status))
                logger.warning(f"[{tk}] {status}")
            if i % 50 == 0:
                logger.info(f"  {i}/{len(tickers)} tickers")
    if errors:
        logger.warning(f"{len(errors)} tickers errored: "
                       f"{[t for t, _ in errors]}")

    parts = [pd.read_parquet(p) for p in sorted(PARTS_DIR.glob("*.parquet"))]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if df.empty:
        logger.warning("No pairs computed — is the 10-K corpus present?")
        return df

    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE, index=False)

    path_counts = df["path"].value_counts().to_dict()
    logger.info(
        f"Saved {len(df)} pairs ({df['ticker'].nunique()} tickers) to "
        f"{CACHE_FILE}. paths={path_counts}. "
        f"sim_minedit mean={df['sim_minedit_10k'].mean():.4f}, "
        f"sim_simple mean={df['sim_simple_10k'].mean():.4f}, "
        f"max|minedit-simple|={df['sim_minedit_10k'].sub(df['sim_simple_10k']).abs().max():.2e}"
    )
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Compute C21/C22 Lazy Prices edit measures on 10-K pairs")
    ap.add_argument("--tickers", nargs="+", help="specific tickers")
    ap.add_argument("--workers", type=int, default=None,
                    help="process-pool size (default: CPUs - 1)")
    args = ap.parse_args()
    df = compute_all(tickers=args.tickers, workers=args.workers)
    if not df.empty:
        print(df[["ticker", "filing_date", "sim_minedit_10k",
                  "sim_simple_10k", "path"]].tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
