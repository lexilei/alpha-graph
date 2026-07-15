"""C24 — 10-Q year-over-year Lazy Prices min-edit similarity + the freshest-
filing combined stream (Cohen-Malloy-Nguyen 2020, "Lazy Prices").

Two pieces, both pinned by the C24 registration (2026-07-15, commit b6bd1f9):

  (a) ``sim_minedit_10q`` — the paper's token-level edit measure on C2's 10-Q
      pair set. Pair ENUMERATION is reused verbatim from ``lazy_prices_10q``
      (``_load_filing_texts(.., "10-Q")``'s amendment / fallback-extraction
      filters + ``_match_yoy``'s same-fiscal-quarter ~365d pairing, which
      controls seasonality — adjacent quarters would not). The similarity ENGINE
      is reused verbatim from C21 ``lazy_prices_edit`` (``difflib.SequenceMatcher
      (autojunk=False).ratio()`` with the length-weighted 10k-token chunked
      fallback above ``FALLBACK_COMBINED_TOKENS``). The tokenizer is PINNED to
      C21's digit-free ``alpha`` mode ``re.findall(r"[a-z]+", text.lower())``
      (pure-digit year/amount churn dropped), matching the digit-free 10-K
      measure the combined stream unions it with.

  (b) ``sim_minedit_fresh`` — the freshest-filing combined stream, built per
      C7's construction (``_load_cos_latest_filing`` in ml_combiner): the union
      of the digit-free 10-K min-edit pairs (``lazy_prices_edit_alpha.parquet``,
      ``sim_minedit_alpha_10k``) with the 10-Q YoY min-edit pairs here, pooled
      into one column, same-day dedup keep-last. This is the 10-K∪10-Q stream
      whose monthly rebalance is effectively a rolling event window — the object
      the C24 freshness-corrected evaluation restricts to fresh names.

Caches:
  data/cache/lazy_prices_edit_10q.parquet    (ticker, filing_date,
                                               prev_filing_date, sim_minedit_10q)
  data/cache/lazy_prices_edit_fresh.parquet   (ticker, filing_date,
                                               sim_minedit_fresh)

Usage:
    python -m alpha_graph.signals.lazy_prices_edit_10q [--tickers AAPL MSFT] \
        [--workers N] [--fresh-only]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR
from alpha_graph.signals.lazy_prices import _load_filing_texts
from alpha_graph.signals.lazy_prices_10q import _match_yoy
from alpha_graph.signals.lazy_prices_edit import (
    FALLBACK_COMBINED_TOKENS,
    pair_similarity,
    tokenize,
)

# Pinned tokenizer mode for C24: C21's digit-free ``[a-z]+`` (the combined
# stream unions this with the digit-free 10-K measure, so both legs must share
# the tokenizer).
TOKENIZER_MODE = "alpha"
MINEDIT_COL = "sim_minedit_10q"

# Per-ticker checkpoint directory + final cache (resume + parallel-safe: each
# worker writes only its own part file).
PARTS_DIR = CACHE_DIR / "lazy_prices_edit_10q_parts"
CACHE_FILE = CACHE_DIR / "lazy_prices_edit_10q.parquet"

# The freshest-filing combined stream (10-K digit-free ∪ 10-Q YoY min-edit).
FRESH_CACHE_FILE = CACHE_DIR / "lazy_prices_edit_fresh.parquet"
K10_ALPHA_CACHE = CACHE_DIR / "lazy_prices_edit_alpha.parquet"
FRESH_COL = "sim_minedit_fresh"


def compute_for_ticker(ticker: str) -> list[dict]:
    """All 10-Q YoY min-edit pairs for one ticker (C2's pair set, C21's engine).

    Pairing is C2's exactly: ``_load_filing_texts(.., "10-Q")`` (amendment /
    fallback filters) then ``_match_yoy`` — for each 10-Q, the prior 10-Q closest
    to ~365 days earlier (same fiscal quarter). The score is the digit-free
    ``alpha``-tokenized SequenceMatcher ratio between that pair, stamped at the
    NEW filing's ``filing_date`` (same availability as C2)."""
    filings = _load_filing_texts(ticker, "10-Q")
    if len(filings) < 4:  # C2 needs >=4 10-Qs for a YoY match
        return []
    dates = [pd.Timestamp(f["filing_date"]) for f in filings]
    # tokenize each filing once — a filing can be both the new (i) and, a year
    # later, the prior (j) member of a pair.
    toks = [tokenize(f["text"], TOKENIZER_MODE) for f in filings]

    rows: list[dict] = []
    for i in range(len(filings)):
        j = _match_yoy(dates, i)
        if j is None:
            continue
        minedit, _simple, path = pair_similarity(toks[j], toks[i])
        rows.append({
            "ticker": ticker,
            "filing_date": filings[i]["filing_date"],
            "prev_filing_date": filings[j]["filing_date"],
            MINEDIT_COL: round(float(minedit), 6),
            "gap_days": (dates[i] - dates[j]).days,   # diagnostics (not cached)
            "path": path,                             # diagnostics (not cached)
        })
    return rows


def _worker(ticker: str) -> tuple[str, str]:
    """Process-pool worker: compute one ticker, checkpoint its part file.
    Top-level + picklable so the spawn start method (macOS default) can
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
    """Compute the 10-Q YoY min-edit similarity for every ticker with a 10-Q
    corpus and cache the result. Parallelized per ticker with a process pool
    (SequenceMatcher is pure-Python CPU work, so processes beat threads);
    resumable via per-ticker part files."""
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
    logger.info(f"Computing 10-Q YoY min-edit ({TOKENIZER_MODE!r} tokenizer) for "
                f"{len(tickers)} tickers on {workers} workers "
                f"(fallback > {FALLBACK_COMBINED_TOKENS:,} tokens)...")

    errors = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (tk, status) in enumerate(ex.map(_worker, tickers), 1):
            if status.startswith("ERROR"):
                errors.append((tk, status))
                logger.warning(f"[{tk}] {status}")
            if i % 50 == 0:
                logger.info(f"  {i}/{len(tickers)} tickers")
    if errors:
        logger.warning(f"{len(errors)} tickers errored: {[t for t, _ in errors]}")

    parts = [pd.read_parquet(p) for p in sorted(PARTS_DIR.glob("*.parquet"))]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if df.empty:
        logger.warning("No 10-Q pairs computed — is the 10-Q corpus present?")
        return df

    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    path_counts = df["path"].value_counts().to_dict()
    logger.info(
        f"10-Q min-edit: {len(df)} pairs ({df['ticker'].nunique()} tickers), "
        f"paths={path_counts}, median gap={df['gap_days'].median():.0f}d, "
        f"{MINEDIT_COL} mean={df[MINEDIT_COL].mean():.4f} "
        f"std={df[MINEDIT_COL].std():.4f}"
    )

    # Cache exactly the four registered columns (gap_days / path were logged).
    out = df[["ticker", "filing_date", "prev_filing_date", MINEDIT_COL]]
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE_FILE, index=False)
    logger.info(f"Saved {len(out)} pairs to {CACHE_FILE}")
    return out


# --------------------------------------------------------------------------- #
# Freshest-filing combined stream (C7's construction on the min-edit measures)
# --------------------------------------------------------------------------- #

def combine_fresh_stream(
    k_df: pd.DataFrame,
    q_df: pd.DataFrame,
    k_col: str = "sim_minedit_alpha_10k",
    q_col: str = MINEDIT_COL,
) -> pd.DataFrame:
    """Union the digit-free 10-K min-edit pairs with the 10-Q YoY min-edit pairs
    into one ``sim_minedit_fresh`` stream, per C7's ``_load_cos_latest_filing``
    construction: pool the raw scores, then same-day dedup keep-last (a ticker
    filing a 10-K and a 10-Q on the same date keeps the 10-Q — appended second,
    so a stable sort makes "last" deterministic). Returns
    ``[ticker, filing_date, sim_minedit_fresh]`` sorted by ticker/date."""
    k = k_df[["ticker", "filing_date", k_col]].rename(columns={k_col: FRESH_COL})
    q = q_df[["ticker", "filing_date", q_col]].rename(columns={q_col: FRESH_COL})
    comb = pd.concat([k, q], ignore_index=True)
    comb["filing_date"] = pd.to_datetime(comb["filing_date"])
    # stable sort so within a (ticker, filing_date) tie the 10-Q (concatenated
    # second) is the "last" row kept — deterministic keep-last.
    comb = comb.sort_values(["ticker", "filing_date"], kind="stable")
    comb = comb.drop_duplicates(["ticker", "filing_date"], keep="last")
    return comb.reset_index(drop=True)


def build_fresh_stream() -> pd.DataFrame:
    """Materialize the freshest-filing combined stream cache from the two
    min-edit pair caches (digit-free 10-K + 10-Q YoY). Needs BOTH."""
    if not (K10_ALPHA_CACHE.exists() and CACHE_FILE.exists()):
        logger.error(
            f"Combined stream needs both {K10_ALPHA_CACHE.name} and "
            f"{CACHE_FILE.name} — run the 10-Q build (and the C21 alpha build) "
            "first."
        )
        return pd.DataFrame()
    k = pd.read_parquet(K10_ALPHA_CACHE)
    q = pd.read_parquet(CACHE_FILE)
    comb = combine_fresh_stream(k, q)
    FRESH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    comb.to_parquet(FRESH_CACHE_FILE, index=False)
    logger.info(
        f"Saved {len(comb)} combined-stream rows ({comb['ticker'].nunique()} "
        f"tickers) to {FRESH_CACHE_FILE}. {FRESH_COL} mean="
        f"{comb[FRESH_COL].mean():.4f}; 10-K rows {len(k)} + 10-Q rows {len(q)} "
        f"- same-day dups {len(k) + len(q) - len(comb)}"
    )
    return comb


def main():
    ap = argparse.ArgumentParser(
        description="Compute 10-Q YoY Lazy Prices min-edit + the combined stream")
    ap.add_argument("--tickers", nargs="+", help="specific tickers")
    ap.add_argument("--workers", type=int, default=None,
                    help="process-pool size (default: CPUs - 1)")
    ap.add_argument("--fresh-only", action="store_true",
                    help="skip the 10-Q build; only (re)build the combined "
                         "stream from existing caches")
    args = ap.parse_args()

    if not args.fresh_only:
        df = compute_all(tickers=args.tickers, workers=args.workers)
        if not df.empty:
            print(df.tail(15).to_string(index=False))
    # the combined stream is a whole-universe artifact — build it only on a full
    # run (no --tickers subset), so a single-ticker debug run never truncates it.
    if args.tickers is None:
        build_fresh_stream()


if __name__ == "__main__":
    main()
