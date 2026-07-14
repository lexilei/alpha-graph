"""Build data/cache/filing_items.parquet: 8-K item codes parsed from filing text.

data/cache/filing_acceptance.parquet is the filing_events base table
(accession -> form/dates/acceptance_ts) but carries no 8-K ITEM CODES, and the
corpus JSONs (data/filings/<TICKER>/8-K_YYYY-MM-DD_ACCESSION.json) store their
text under sections={"full_text": ...} — item labels exist only as "Item X.XX"
headers inside the text. C17 sue_pead already recovered Item 2.02 this way
(ITEM_202_RE = r"item\\s+2\\.02", searched anywhere in the joined section text);
this script generalizes that extraction to ALL 8-K item codes.

EXTRACTION (pinned, tests/test_filing_items.py): a code is credited when an
"Item<ws>#.##" header STARTS a line —

    ^\\s*item\\s+([1-9]\\.\\d{2})(?!\\d)   IGNORECASE | MULTILINE

- token core `item\\s+` is exactly C17's ITEM_202_RE with the code generalized
  to the 8-K #.## family (items 1.01-9.01); `(?!\\d)` right-bounds the code;
- caps ("ITEM 2.02"), trailing dot, "(b)" suffixes, and unicode spaces
  (\\s matches NBSP etc.) all match;
- the LINE-START ANCHOR is the one tightening over C17: mid-line mentions
  ("pursuant to Item 2.02 above", "disclosures required by Item 1.01") do NOT
  count. The 2.02 delta this creates vs the C17 anywhere-search is quantified
  in the build-time reconciliation below (anchored is a strict subset).
- NOT matched, by design: pre-2004 numbering ("Item 5. Other Events" — moot,
  corpus starts 2011), Regulation S-K/exhibit refs ("Item 601" — not #.##),
  table-mangled headers whose code sits in another cell ("Item    Financial"),
  and glued "Item8.01" with no whitespace (out of the C17 token family;
  ~0.3% of files, counted in the zero-item characterization).

Output data/cache/filing_items.parquet — one row per corpus 8-K file
(zero-item rows kept; same-accession rows can repeat under dual-listed
ticker directories):
    accession    digits-only accession number (str, = filename; join key to
                 filing_acceptance.parquet, which is NOT modified)
    ticker       corpus directory ticker (str)
    filing_date  official filing date from the filename (datetime, midnight)
    items        comma-joined sorted unique codes, e.g. "2.02,9.01" ("" if none)
    n_items      size of that set (int)

Usage:
    python scripts/build_filing_items.py
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR
from alpha_graph.signals.sue_pead import ITEM_202_RE

# C17's `item\s+2\.02` with the code generalized to #.## and a line-start pin.
ITEM_HEADER_RE = re.compile(r"^\s*item\s+([1-9]\.\d{2})(?!\d)",
                            re.IGNORECASE | re.MULTILINE)
# Zero-item diagnosis only (never feeds the items column):
GLUED_RE = re.compile(r"^\s*item([1-9]\.\d{2})(?!\d)", re.IGNORECASE | re.MULTILINE)
ANYWHERE_RE = re.compile(r"item\s+([1-9]\.\d{2})(?!\d)", re.IGNORECASE)

C17_202_COUNT = 30_408        # scan_8k_202's Item-2.02 tally, pinned 2026-07-13
CACHE_NAME = "filing_items.parquet"


def extract_items(text: str) -> list[str]:
    """Sorted unique 8-K item codes whose header starts a line of `text`."""
    return sorted(set(ITEM_HEADER_RE.findall(text)))


def diagnose_zero(text: str) -> str:
    """Why a filing yielded no items (mutually exclusive, first match wins)."""
    if GLUED_RE.search(text):
        return "glued_header"          # "Item8.01" — no whitespace after Item
    if ANYWHERE_RE.search(text):
        return "midline_mention_only"  # body-text refs, never a line-start header
    if "item" in text.lower():
        return "item_word_no_code"     # table-mangled header / 10-K-style refs
    return "no_item_text"


def scan_corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk every 8-K JSON (same walk as C17's scan_8k_202).

    Returns (df, zeros): df has the output columns plus `c17_has_202`
    (ITEM_202_RE searched anywhere — reconciliation only, dropped before
    save); zeros characterizes the zero-item files.
    """
    rows, zero_rows = [], []
    for d in sorted(FILINGS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        for f in sorted(d.glob("8-K_*.json")):
            parts = f.name.split("_")
            if len(parts) < 3:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.debug(f"[{d.name}] failed to parse {f.name}")
                continue
            text = "\n\n".join(data.get("sections", {}).values())
            items = extract_items(text)
            accession = parts[2].removesuffix(".json")
            rows.append((accession, d.name, parts[1], ",".join(items),
                         len(items), bool(ITEM_202_RE.search(text))))
            if not items:
                zero_rows.append((accession, d.name, parts[1],
                                  diagnose_zero(text), len(text)))
    df = pd.DataFrame(rows, columns=["accession", "ticker", "filing_date",
                                     "items", "n_items", "c17_has_202"])
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    n_bad = int(df["filing_date"].isna().sum())
    if n_bad:
        logger.warning(f"dropped {n_bad} rows with unparseable filename dates")
        df = df.dropna(subset=["filing_date"])
    zeros = pd.DataFrame(zero_rows, columns=["accession", "ticker",
                                             "filing_date", "why", "text_len"])
    return df.reset_index(drop=True), zeros


def validate(df: pd.DataFrame, zeros: pd.DataFrame) -> None:
    """Build-time checks, printed (no file output)."""
    n = len(df)
    n_hit = int((df["n_items"] > 0).sum())
    print(f"\n8-K corpus files scanned: {n}")
    print(f"coverage: {n_hit}/{n} = {n_hit / n:.2%} with >=1 item parsed")

    dup = int(df["accession"].duplicated().sum())
    print(f"duplicate accessions (dual-listed ticker dirs): {dup}")

    print("\n--- zero-item characterization ---")
    if zeros.empty:
        print("(none)")
    else:
        print(zeros["why"].value_counts().to_string())
        print(f"zero-item text length: median {int(zeros['text_len'].median())}, "
              f"max {int(zeros['text_len'].max())}")
        print(zeros.groupby("why", sort=False).head(2).to_string(index=False))

    print("\n--- item frequency (files containing the item) ---")
    freq = (df.loc[df["n_items"] > 0, "items"].str.split(",")
            .explode().value_counts())
    print(freq.head(15).to_string())

    print("\n--- items-per-filing distribution ---")
    print(df["n_items"].value_counts().sort_index().to_string())
    print(f"p50 items per filing: {df['n_items'].median():.0f} "
          f"(mean {df['n_items'].mean():.2f})")

    print("\n--- Item 2.02 reconciliation vs C17 sue_pead ---")
    c17 = int(df["c17_has_202"].sum())
    has202 = df["items"].str.split(",").apply(lambda s: "2.02" in s)
    mine = int(has202.sum())
    only_c17 = int((df["c17_has_202"] & ~has202).sum())
    only_mine = int((~df["c17_has_202"] & has202).sum())
    print(f"C17 ITEM_202_RE anywhere-search: {c17} "
          f"(scan_8k_202 pinned {C17_202_COUNT}: "
          f"{'MATCH' if c17 == C17_202_COUNT else 'MISMATCH'})")
    print(f"header-anchored 2.02 (this table): {mine}")
    print(f"  in C17 set only (2.02 mentioned mid-line, never a line-start "
          f"header): {only_c17}")
    print(f"  in this table only (must be 0 — anchored is a subset): {only_mine}")

    join = pd.read_parquet(CACHE_DIR / "filing_acceptance.parquet")
    in_acc = df["accession"].isin(set(join["accession"])).mean()
    print(f"\naccessions found in filing_acceptance.parquet: {in_acc:.2%}")


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    df, zeros = scan_corpus()
    logger.info(f"scanned {len(df)} 8-K filings, "
                f"{int((df['n_items'] > 0).sum())} with >=1 item "
                f"({df['ticker'].nunique()} tickers)")
    out = df[["accession", "ticker", "filing_date", "items", "n_items"]]
    path = CACHE_DIR / CACHE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    logger.info(f"Saved {len(out)} rows to {path}")
    validate(df, zeros)


if __name__ == "__main__":
    main()
