"""Fundamental data pipeline — quality and value factors via yfinance.

Downloads key fundamental ratios for the stock universe to use as
features in the ML combiner. Factors are based on established academic
literature (Fama-French, Novy-Marx, Asness):

Quality factors:
  - roe: Return on Equity (net income / shareholders equity)
  - gross_margin: Gross profit / Revenue
  - debt_equity: Total debt / Shareholders equity

Value factors:
  - pe_trailing: Price / Trailing 12-month EPS
  - pb_ratio: Price / Book value per share
  - div_yield: Annual dividend / Price

All factors are cross-sectionally ranked (0-1) so the ML model sees
relative cheapness/quality rather than raw levels. This avoids stale
absolute values dominating when fundamentals update quarterly.

Usage:
    python -m alpha_graph.data.fundamentals [--tickers AAPL MSFT] [--max-tickers 100]
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from alpha_graph.config import CACHE_DIR, cfg


# Fields we pull from yfinance .info dict, mapped to our column names
YF_FIELD_MAP = {
    # Value
    "trailingPE": "pe_trailing",
    "priceToBook": "pb_ratio",
    "dividendYield": "div_yield",
    # Quality
    "returnOnEquity": "roe",
    "grossMargins": "gross_margin",
    "debtToEquity": "debt_equity",
}

FUNDAMENTAL_COLS = list(YF_FIELD_MAP.values())


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Fetch fundamental ratios for a list of tickers via yfinance.

    Returns DataFrame with: ticker, date, pe_trailing, pb_ratio,
    div_yield, roe, gross_margin, debt_equity.

    Uses yfinance Ticker.info which returns the latest available data.
    Failures for individual tickers are logged and skipped.
    """
    logger.info(f"Fetching fundamentals for {len(tickers)} tickers...")
    now = datetime.now().strftime("%Y-%m-%d")

    rows = []
    failed = []
    for i, ticker in enumerate(tickers, 1):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}

            row = {"ticker": ticker, "date": now}
            for yf_key, col_name in YF_FIELD_MAP.items():
                val = info.get(yf_key)
                if val is not None:
                    row[col_name] = float(val)
                else:
                    row[col_name] = np.nan

            rows.append(row)
        except Exception as e:
            logger.debug(f"[{ticker}] fundamentals fetch failed: {e}")
            failed.append(ticker)

        if i % 50 == 0:
            logger.info(f"  Progress: {i}/{len(tickers)}")

    if failed:
        logger.warning(f"Failed to fetch {len(failed)} tickers: {failed[:10]}...")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])

    # Winsorize extremes (P/E > 500 or < 0 is noise)
    if "pe_trailing" in df.columns:
        df.loc[df["pe_trailing"] < 0, "pe_trailing"] = np.nan
        df.loc[df["pe_trailing"] > 500, "pe_trailing"] = np.nan
    if "debt_equity" in df.columns:
        df.loc[df["debt_equity"] < 0, "debt_equity"] = np.nan
        df.loc[df["debt_equity"] > 1000, "debt_equity"] = np.nan

    logger.info(
        f"Fetched fundamentals: {len(df)} tickers, "
        f"{df[FUNDAMENTAL_COLS].notna().mean().mean():.0%} avg coverage"
    )

    return df


def rank_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally rank fundamental factors to [0, 1].

    Ranking avoids issues with stale absolute values and makes factors
    comparable across time periods. Higher rank = higher raw value.

    For value factors (PE, PB): low rank = cheap = good.
    For quality factors (ROE, margin): high rank = good.
    The ML model learns the optimal direction.
    """
    df = df.copy()
    for col in FUNDAMENTAL_COLS:
        if col in df.columns:
            ranked_col = f"{col}_rank"
            df[ranked_col] = df[col].rank(pct=True)
    return df


def download_and_cache(
    tickers: list[str] | None = None,
    max_tickers: int | None = None,
) -> pd.DataFrame:
    """Fetch fundamentals, rank, and cache to parquet."""
    if tickers is None:
        from alpha_graph.data.universe import get_sp500_tickers
        tickers = get_sp500_tickers(max_tickers=max_tickers)

    df = fetch_fundamentals(tickers)
    if df.empty:
        return df

    df = rank_fundamentals(df)

    out_path = CACHE_DIR / "fundamentals.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Download fundamental data")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--max-tickers", type=int, default=100)
    args = parser.parse_args()

    df = download_and_cache(tickers=args.tickers, max_tickers=args.max_tickers)
    if not df.empty:
        print(f"\nFundamentals for {len(df)} tickers:")
        print(f"\nCoverage:")
        for col in FUNDAMENTAL_COLS:
            pct = df[col].notna().mean() * 100
            print(f"  {col:>15s}: {pct:5.1f}%")

        print(f"\nSample (top 10 by ROE):")
        top = df.dropna(subset=["roe"]).nlargest(10, "roe")
        print(top[["ticker"] + FUNDAMENTAL_COLS].to_string(index=False))


if __name__ == "__main__":
    main()
