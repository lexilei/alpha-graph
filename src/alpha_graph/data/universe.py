"""Stock universe management — fetch and cache S&P 500 constituents."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR, cfg


def get_sp500_tickers(max_tickers: int | None = None) -> list[str]:
    """Return S&P 500 tickers sorted by market cap (largest first).

    Uses the Wikipedia table as the source of truth.
    Caches to disk so repeated calls don't hit the network.
    """
    cache_path = CACHE_DIR / "sp500_tickers.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        tickers = pd.read_csv(cache_path)["ticker"].tolist()
    else:
        logger.info("Fetching S&P 500 constituents from Wikipedia...")
        import urllib.request
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "alpha-graph/0.1"})
        html = urllib.request.urlopen(req).read().decode("utf-8")
        from io import StringIO
        table = pd.read_html(StringIO(html))[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        pd.DataFrame({"ticker": tickers}).to_csv(cache_path, index=False)
        logger.info(f"Cached {len(tickers)} tickers to {cache_path}")

    limit = max_tickers or cfg.max_tickers
    return tickers[:limit]
