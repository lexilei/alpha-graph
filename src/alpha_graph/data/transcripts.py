"""Earnings call transcript collector using Finnhub API.

Downloads earnings call transcripts and extracts key metrics
(executive communication patterns, sentiment) for signal generation.

Usage:
    python -m alpha_graph.data.transcripts [--tickers AAPL MSFT] [--max-tickers 50]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from loguru import logger

from alpha_graph.config import TRANSCRIPTS_DIR, cfg

# Finnhub free tier: 60 calls/minute
FINNHUB_REQUEST_DELAY = 1.1  # ~55 req/min to stay safe


def _get_client():
    """Initialize Finnhub client."""
    import finnhub

    if not cfg.finnhub_api_key:
        raise ValueError("FINNHUB_API_KEY not set. Add it to .env")
    return finnhub.Client(api_key=cfg.finnhub_api_key)


def get_earnings_calendar(
    client, ticker: str, from_date: str | None = None, to_date: str | None = None
) -> list[dict]:
    """Get earnings dates for a ticker."""
    if from_date is None:
        from_date = f"{datetime.now().year - cfg.filing_years_back}-01-01"
    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d")

    try:
        calendar = client.earnings_calendar(
            _from=from_date, to=to_date, symbol=ticker
        )
        return calendar.get("earningsCalendar", [])
    except Exception as e:
        logger.warning(f"[{ticker}] Earnings calendar failed: {e}")
        return []


def download_transcript(client, ticker: str, year: int, quarter: int) -> dict | None:
    """Download a single earnings call transcript."""
    try:
        transcript = client.earnings_call_transcripts(
            symbol=ticker, year=year, quarter=quarter
        )
        if not transcript or not transcript.get("transcript"):
            return None
        return transcript
    except Exception as e:
        logger.debug(f"[{ticker}] Transcript {year}Q{quarter} not available: {e}")
        return None


def _format_transcript(raw: dict) -> dict:
    """Extract structured data from a raw Finnhub transcript response."""
    entries = raw.get("transcript", [])
    full_text_parts = []
    speakers = {}

    for entry in entries:
        name = entry.get("name", "Unknown")
        speech = entry.get("speech", [])
        text = " ".join(speech) if isinstance(speech, list) else str(speech)
        full_text_parts.append(f"{name}: {text}")

        if name not in speakers:
            speakers[name] = {"name": name, "word_count": 0, "segments": 0}
        speakers[name]["word_count"] += len(text.split())
        speakers[name]["segments"] += 1

    return {
        "symbol": raw.get("symbol", ""),
        "year": raw.get("year"),
        "quarter": raw.get("quarter"),
        "full_text": "\n\n".join(full_text_parts),
        "speakers": list(speakers.values()),
        "total_words": sum(s["word_count"] for s in speakers.values()),
    }


def download_transcripts_for_ticker(
    ticker: str,
    years_back: int | None = None,
) -> list[dict]:
    """Download all available transcripts for a ticker."""
    client = _get_client()
    years_back = years_back or cfg.filing_years_back
    current_year = datetime.now().year

    ticker_dir = TRANSCRIPTS_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for year in range(current_year - years_back, current_year + 1):
        for quarter in range(1, 5):
            out_path = ticker_dir / f"earnings_{year}_Q{quarter}.json"

            if out_path.exists():
                saved.append({"ticker": ticker, "year": year, "quarter": quarter})
                continue

            raw = download_transcript(client, ticker, year, quarter)
            time.sleep(FINNHUB_REQUEST_DELAY)

            if raw is None:
                continue

            record = _format_transcript(raw)
            if record["total_words"] < 50:
                continue

            out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            saved.append({"ticker": ticker, "year": year, "quarter": quarter})
            logger.info(
                f"[{ticker}] Saved {year} Q{quarter} transcript "
                f"({record['total_words']} words, {len(record['speakers'])} speakers)"
            )

    return saved


def download_all(
    tickers: list[str] | None = None,
    max_tickers: int | None = None,
    years_back: int | None = None,
) -> None:
    """Download transcripts for multiple tickers."""
    if tickers is None:
        from alpha_graph.data.universe import get_sp500_tickers

        tickers = get_sp500_tickers(max_tickers=max_tickers)

    logger.info(f"Downloading earnings transcripts for {len(tickers)} tickers...")
    total = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] Processing {ticker}...")
        results = download_transcripts_for_ticker(ticker, years_back=years_back)
        total += len(results)

    logger.info(f"Done. {total} transcripts saved to {TRANSCRIPTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Download earnings call transcripts")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--max-tickers", type=int, default=100)
    parser.add_argument("--years-back", type=int, default=5)
    args = parser.parse_args()

    download_all(
        tickers=args.tickers,
        max_tickers=args.max_tickers,
        years_back=args.years_back,
    )


if __name__ == "__main__":
    main()
