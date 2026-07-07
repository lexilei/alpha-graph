"""Factors 2/3 — 8-K item-type prior scores (see ITEM_SCORES), exponentially
weighted per ticker (lambda=0.9/month).

Usage:
    python -m alpha_graph.signals.event_signal [--tickers AAPL MSFT] [--timeseries]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, FILINGS_DIR


# Prior score for each 8-K item type based on empirical market impact
ITEM_SCORES: dict[str, float] = {
    "1.01": 0.3,    # Entry into a Material Definitive Agreement
    "1.02": -0.5,   # Termination of a Material Definitive Agreement
    "1.03": -0.2,   # Bankruptcy or Receivership
    "1.04": -0.3,   # Mine Safety
    "2.01": 0.2,    # Completion of Acquisition or Disposition of Assets
    "2.02": 0.1,    # Results of Operations and Financial Condition
    "2.03": -0.1,   # Creation of a Direct Financial Obligation
    "2.04": -0.4,   # Triggering Events (Accelerated Obligation)
    "2.05": -0.4,   # Costs Associated with Exit or Disposal Activities
    "2.06": -0.6,   # Material Impairments
    "3.01": -0.1,   # Notice of Delisting
    "3.02": 0.0,    # Unregistered Sales of Equity Securities
    "3.03": -0.1,   # Material Modification of Rights of Security Holders
    "4.01": -0.7,   # Changes in Registrant's Certifying Accountant
    "4.02": -0.8,   # Non-Reliance on Previously Issued Financial Statements
    "5.01": 0.0,    # Changes in Control of Registrant
    "5.02": -0.3,   # Departure of Directors or Certain Officers
    "5.03": 0.0,    # Amendments to Articles of Incorporation
    "5.04": 0.0,    # Temporary Suspension of Trading
    "5.05": 0.0,    # Amendments to the Code of Ethics
    "5.06": 0.0,    # Change in Shell Company Status
    "5.07": 0.0,    # Submission of Matters to a Vote
    "5.08": 0.0,    # Shareholder Nominations
    "7.01": 0.0,    # Regulation FD Disclosure
    "8.01": 0.0,    # Other Events
    "9.01": 0.0,    # Financial Statements and Exhibits
}

# Exponential decay parameter: weight = lambda^(months_ago)
DECAY_LAMBDA = 0.9


def _extract_item_numbers(text: str) -> list[str]:
    """Extract 8-K item numbers from filing text or section keys.

    Looks for patterns like "Item 1.01", "Item 2.05", "ITEM 4.01", etc.
    Returns deduplicated list of item number strings like ["1.01", "2.05"].
    """
    pattern = r"item\s+(\d+\.\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return list(set(matches))


def _load_8k_filings(ticker: str) -> list[dict]:
    """Load all 8-K filings for a ticker, sorted chronologically.

    Returns list of dicts with: filing_date, items, text, accession_number.
    """
    ticker_dir = FILINGS_DIR / ticker
    if not ticker_dir.exists():
        return []

    filings = []
    for f in sorted(ticker_dir.glob("8-K_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug(f"[{ticker}] Failed to parse {f.name}")
            continue

        sections = data.get("sections", {})
        full_text = "\n\n".join(sections.values())

        # Extract item numbers from section keys and text
        items_from_keys = []
        for key in sections:
            items_from_keys.extend(_extract_item_numbers(key))
        items_from_text = _extract_item_numbers(full_text)

        all_items = list(set(items_from_keys + items_from_text))

        filing_date = data.get("filing_date", "")
        if not filing_date:
            continue

        filings.append({
            "filing_date": filing_date,
            "items": all_items,
            "text": full_text,
            "accession_number": data.get("accession_number", ""),
            "filename": f.name,
        })

    return filings


def score_filing(items: list[str]) -> float:
    """Score a single 8-K filing based on its item types.

    If multiple items are present, return the most extreme score
    (furthest from zero). This captures the dominant signal.
    """
    if not items:
        return 0.0

    scores = [ITEM_SCORES.get(item, 0.0) for item in items]
    if not scores:
        return 0.0

    # Return the score with largest absolute value (most material event)
    return max(scores, key=abs)


def compute_event_signal(
    ticker: str,
    as_of_date: str | pd.Timestamp | None = None,
    lookback_months: int = 12,
) -> dict:
    """Compute event signal for a single ticker.

    Uses exponentially-weighted average of event scores over the lookback window.
    More recent events get higher weight (lambda^months_ago decay).

    Args:
        ticker: stock ticker symbol
        as_of_date: compute signal as of this date (default: now)
        lookback_months: how far back to look for events

    Returns:
        Dict with: ticker, event_score, event_count, last_event_date,
                   most_negative_event, most_positive_event
    """
    filings = _load_8k_filings(ticker)

    if not filings:
        return {
            "ticker": ticker,
            "event_score": 0.0,
            "event_count": 0,
            "last_event_date": None,
            "most_negative_event": None,
            "most_positive_event": None,
        }

    # Parse dates
    if as_of_date is None:
        as_of_date = pd.Timestamp.now()
    else:
        as_of_date = pd.Timestamp(as_of_date)

    cutoff_date = as_of_date - pd.DateOffset(months=lookback_months)

    # Filter to lookback window and score each filing
    scored_events = []
    for filing in filings:
        fdate = pd.Timestamp(filing["filing_date"])
        if fdate > as_of_date or fdate < cutoff_date:
            continue

        event_score = score_filing(filing["items"])
        months_ago = max(0.0, (as_of_date - fdate).days / 30.44)

        scored_events.append({
            "filing_date": fdate,
            "items": filing["items"],
            "event_score": event_score,
            "months_ago": months_ago,
            "weight": DECAY_LAMBDA ** months_ago,
        })

    if not scored_events:
        return {
            "ticker": ticker,
            "event_score": 0.0,
            "event_count": 0,
            "last_event_date": None,
            "most_negative_event": None,
            "most_positive_event": None,
        }

    # Exponentially-weighted average
    total_weight = sum(e["weight"] for e in scored_events)
    weighted_score = sum(e["event_score"] * e["weight"] for e in scored_events)
    event_signal = weighted_score / total_weight if total_weight > 0 else 0.0

    # Identify extremes
    most_neg = min(scored_events, key=lambda e: e["event_score"])
    most_pos = max(scored_events, key=lambda e: e["event_score"])

    return {
        "ticker": ticker,
        "event_score": round(float(event_signal), 6),
        "event_count": len(scored_events),
        "last_event_date": max(e["filing_date"] for e in scored_events).strftime("%Y-%m-%d"),
        "most_negative_event": (
            f"Item {','.join(most_neg['items'])} ({most_neg['filing_date'].strftime('%Y-%m-%d')}): "
            f"score={most_neg['event_score']:.2f}"
        ) if most_neg["event_score"] < 0 else None,
        "most_positive_event": (
            f"Item {','.join(most_pos['items'])} ({most_pos['filing_date'].strftime('%Y-%m-%d')}): "
            f"score={most_pos['event_score']:.2f}"
        ) if most_pos["event_score"] > 0 else None,
    }


def compute_all(
    tickers: list[str] | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute 8-K event signals for all tickers.

    Returns DataFrame with columns: ticker, event_score, event_count,
    last_event_date, most_negative_event, most_positive_event.
    """
    if tickers is None:
        if not FILINGS_DIR.exists():
            logger.error(f"No filings directory at {FILINGS_DIR}. Run filings download first.")
            return pd.DataFrame()
        tickers = sorted(
            d.name for d in FILINGS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    logger.info(f"Computing 8-K event signals for {len(tickers)} tickers...")
    rows = []

    for ticker in tickers:
        result = compute_event_signal(ticker, as_of_date=as_of_date)
        rows.append(result)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No event signals computed")
        return df

    # Sort by event_score to show most extreme signals
    df = df.sort_values("event_score").reset_index(drop=True)

    # Save to cache
    out_path = CACHE_DIR / "event_signals.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} event signals to {out_path}")

    # Summary stats
    has_events = df[df["event_count"] > 0]
    logger.info(
        f"Event signal stats: {len(has_events)}/{len(df)} tickers have events, "
        f"mean_score={df['event_score'].mean():.4f}, "
        f"std={df['event_score'].std():.4f}, "
        f"min={df['event_score'].min():.4f}, "
        f"max={df['event_score'].max():.4f}"
    )

    return df


def compute_time_series(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    freq: str = "M",
) -> pd.DataFrame:
    """Compute event signal time series for walk-forward backtesting.

    For each month-end in the date range, computes the event signal for
    each ticker as of that date. This avoids lookahead bias.

    Returns DataFrame with columns: date, ticker, event_score, event_count.
    """
    if tickers is None:
        if not FILINGS_DIR.exists():
            return pd.DataFrame()
        tickers = sorted(
            d.name for d in FILINGS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    if start_date is None:
        start_date = "2023-01-31"
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    dates = pd.date_range(start=start_date, end=end_date, freq="ME")
    logger.info(
        f"Computing event signal time series: {len(tickers)} tickers x {len(dates)} dates"
    )

    rows = []
    for dt in dates:
        for ticker in tickers:
            result = compute_event_signal(ticker, as_of_date=dt)
            result["date"] = dt.strftime("%Y-%m-%d")
            rows.append(result)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        out_path = CACHE_DIR / "event_signals_timeseries.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df)} event signal observations to {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Compute 8-K event-driven signals")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--as-of", help="Compute signals as of this date (YYYY-MM-DD)")
    parser.add_argument(
        "--timeseries", action="store_true",
        help="Compute full time series for backtesting"
    )
    args = parser.parse_args()

    if args.timeseries:
        df = compute_time_series(tickers=args.tickers)
    else:
        df = compute_all(tickers=args.tickers, as_of_date=args.as_of)

    if not df.empty:
        print("\n--- 8-K Event Signals ---")
        display_cols = ["ticker", "event_score", "event_count", "last_event_date"]
        if "date" in df.columns:
            display_cols = ["date"] + display_cols
        print(df[display_cols].to_string())


if __name__ == "__main__":
    main()
