"""SEC EDGAR filing downloader using edgartools.

Downloads 10-K and 10-Q filings for S&P 500 companies and extracts
key sections (Risk Factors, MD&A) as plain text for downstream analysis.

Usage:
    python -m alpha_graph.data.filings [--tickers AAPL MSFT] [--max-tickers 50] [--years-back 3]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from alpha_graph.config import FILINGS_DIR, cfg

# Rate limit: SEC asks for max 10 requests/second
SEC_REQUEST_DELAY = 0.12  # ~8 req/s to stay safe


def _init_edgar() -> None:
    """Set the SEC EDGAR identity from config."""
    from edgar import set_identity

    if not cfg.sec_user_agent:
        raise ValueError(
            "SEC_EDGAR_USER_AGENT not set. Add it to .env "
            '(format: "FirstName LastName your@email.com")'
        )
    set_identity(cfg.sec_user_agent)


def _extract_10k_sections(filing) -> dict[str, str]:
    """Extract key sections from a 10-K filing."""
    try:
        ten_k = filing.obj()
    except Exception as e:
        logger.warning(f"Could not parse 10-K structure: {e}")
        return {"full_text": filing.text()[:500_000]}

    sections = {}
    for attr, label in [
        ("business", "Item 1 - Business"),
        ("risk_factors", "Item 1A - Risk Factors"),
        ("management_discussion", "Item 7 - MD&A"),
    ]:
        try:
            text = getattr(ten_k, attr, None)
            if text:
                sections[label] = str(text)
        except Exception:
            pass

    if not sections:
        try:
            sections["full_text"] = filing.text()[:500_000]
        except Exception:
            pass
    return sections


def _extract_10q_sections(filing) -> dict[str, str]:
    """Extract key sections from a 10-Q filing."""
    try:
        ten_q = filing.obj()
    except Exception as e:
        logger.warning(f"Could not parse 10-Q structure: {e}")
        try:
            return {"full_text": filing.text()[:500_000]}
        except Exception:
            return {}

    sections = {}
    # 10-Q items are part-qualified
    for key, label in [
        ("Part I, Item 2", "MD&A"),
        ("Part II, Item 1A", "Risk Factors"),
    ]:
        try:
            text = ten_q[key]
            if text:
                sections[label] = str(text)
        except Exception:
            pass

    if not sections:
        try:
            sections["full_text"] = filing.text()[:500_000]
        except Exception:
            pass
    return sections


def _extract_8k_sections(filing) -> dict[str, str]:
    """Extract text from an 8-K filing (current reports / material events)."""
    try:
        text = filing.text()[:200_000]
        if text:
            return {"full_text": text}
    except Exception as e:
        logger.warning(f"Could not extract 8-K text: {e}")
    return {}


def download_filings_for_ticker(
    ticker: str,
    form_types: list[str] | None = None,
    years_back: int | None = None,
) -> list[dict]:
    """Download filings for a single ticker and save extracted sections to disk.

    Returns list of metadata dicts for each filing saved.
    """
    from edgar import Company

    form_types = form_types or cfg.filing_types
    years_back = years_back or cfg.filing_years_back
    current_year = datetime.now().year
    years = list(range(current_year - years_back, current_year + 1))

    ticker_dir = FILINGS_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    try:
        company = Company(ticker)
    except Exception as e:
        logger.error(f"[{ticker}] Company lookup failed: {e}")
        return saved

    for form_type in form_types:
        try:
            filings = company.get_filings(form=form_type, year=years)
        except Exception as e:
            logger.warning(f"[{ticker}] Failed to get {form_type} filings: {e}")
            continue

        if filings is None:
            continue

        for i in range(len(filings)):
            try:
                filing = filings[i]
            except (IndexError, Exception):
                break

            accession = filing.accession_number.replace("-", "")
            filing_date = str(filing.filing_date)
            out_path = ticker_dir / f"{form_type}_{filing_date}_{accession}.json"

            if out_path.exists():
                logger.debug(f"[{ticker}] Skipping existing {out_path.name}")
                saved.append({"ticker": ticker, "form": form_type, "date": filing_date})
                continue

            # Extract sections
            if form_type == "10-K":
                sections = _extract_10k_sections(filing)
            elif form_type == "8-K":
                sections = _extract_8k_sections(filing)
            else:
                sections = _extract_10q_sections(filing)

            record = {
                "ticker": ticker,
                "company_name": company.name,
                "form_type": form_type,
                "filing_date": filing_date,
                "accession_number": filing.accession_number,
                "sections": sections,
            }

            out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            saved.append({"ticker": ticker, "form": form_type, "date": filing_date})
            logger.info(f"[{ticker}] Saved {form_type} {filing_date}")

            time.sleep(SEC_REQUEST_DELAY)

    return saved


def download_all(
    tickers: list[str] | None = None,
    max_tickers: int | None = None,
    years_back: int | None = None,
) -> None:
    """Download filings for multiple tickers."""
    _init_edgar()

    if tickers is None:
        from alpha_graph.data.universe import get_sp500_tickers

        tickers = get_sp500_tickers(max_tickers=max_tickers)

    logger.info(f"Downloading filings for {len(tickers)} tickers...")
    total_saved = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] Processing {ticker}...")
        results = download_filings_for_ticker(ticker, years_back=years_back)
        total_saved += len(results)
        time.sleep(SEC_REQUEST_DELAY)

    logger.info(f"Done. {total_saved} filings saved to {FILINGS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Download SEC filings")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to download")
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
