"""Market data pipeline — price and return data via yfinance.

Downloads OHLCV data for the stock universe to compute forward returns
for signal evaluation and backtesting.

Usage:
    python -m alpha_graph.data.market [--tickers AAPL MSFT] [--years-back 5]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger

from alpha_graph.config import CACHE_DIR, cfg


def download_prices(
    tickers: list[str],
    years_back: int | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data for a list of tickers.

    Returns a DataFrame with columns: ticker, date, open, high, low, close,
    adj_close, volume.
    """
    years_back = years_back or cfg.filing_years_back
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    start = end - pd.DateOffset(years=years_back)

    logger.info(f"Downloading prices for {len(tickers)} tickers ({start.date()} to {end.date()})...")

    # yfinance supports batch downloads
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        logger.error("No price data returned")
        return pd.DataFrame()

    # Handle single vs multi-ticker format
    if len(tickers) == 1:
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

    # Reshape to long format
    records = []
    for ticker in tickers:
        try:
            df_t = raw.xs(ticker, axis=1, level=1) if len(tickers) > 1 else raw[[(c, ticker) for c in raw.columns.get_level_values(0).unique()]]
            if len(tickers) == 1:
                df_t.columns = df_t.columns.get_level_values(0)
            df_t = df_t.dropna(subset=["Close"])
            if df_t.empty:
                continue
            df_t = df_t.reset_index()
            df_t.columns = [c.lower().replace(" ", "_") for c in df_t.columns]
            df_t["ticker"] = ticker
            records.append(df_t)
        except Exception as e:
            logger.debug(f"[{ticker}] Price data extraction failed: {e}")

    if not records:
        return pd.DataFrame()

    prices = pd.concat(records, ignore_index=True)
    prices = prices.rename(columns={"date": "date"})
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add forward return columns to price data.

    Adds: ret_1d, ret_5d, ret_21d (1-day, 1-week, 1-month forward returns).
    These are the target variables for signal evaluation.
    """
    if prices.empty:
        return prices

    prices = prices.copy()
    prices = prices.sort_values(["ticker", "date"])

    for horizon, col in [(1, "ret_1d"), (5, "ret_5d"), (21, "ret_21d")]:
        prices[col] = prices.groupby("ticker")["close"].transform(
            lambda s: s.pct_change(horizon).shift(-horizon)
        )

    return prices


def download_and_cache(
    tickers: list[str] | None = None,
    max_tickers: int | None = None,
    years_back: int | None = None,
) -> pd.DataFrame:
    """Download prices, compute returns, and cache to parquet."""
    if tickers is None:
        from alpha_graph.data.universe import get_sp500_tickers

        tickers = get_sp500_tickers(max_tickers=max_tickers)

    prices = download_prices(tickers, years_back=years_back)
    if prices.empty:
        return prices

    prices = compute_returns(prices)

    out_path = CACHE_DIR / "market_data.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(prices)} rows to {out_path}")

    return prices


def main():
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--max-tickers", type=int, default=100)
    parser.add_argument("--years-back", type=int, default=5)
    args = parser.parse_args()

    df = download_and_cache(
        tickers=args.tickers,
        max_tickers=args.max_tickers,
        years_back=args.years_back,
    )
    if not df.empty:
        print(f"\nDownloaded {df['ticker'].nunique()} tickers, {len(df)} rows")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
