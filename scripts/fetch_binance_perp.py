"""Download Binance USDT-M perpetual daily klines + funding rates.

Source: data.binance.vision public S3 bucket (monthly zips). The bucket
listing includes delisted symbols, so the universe is survivorship-free as
of each symbol's listing window. USDT-margined symbols only; USDC/BUSD
duplicates of the same underlying are skipped.

Output:
  data/raw/binance/perp_klines_1d.parquet   (symbol, date, open, high, low,
                                             close, volume, quote_volume, n_trades)
  data/raw/binance/perp_funding.parquet     (symbol, funding_time, funding_rate)

Usage: .venv/bin/python scripts/fetch_binance_perp.py
"""

from __future__ import annotations

import csv
import io
import re
import sys
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests

_tls = threading.local()

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "binance"
N_WORKERS = 24

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_volume", "taker_buy_quote_volume",
    "ignore",
]


def _get(url: str, retries: int = 4) -> bytes:
    """Keep-alive GET: one requests.Session per thread (67k tiny files —
    per-request TLS handshakes were 10x slower than the payloads)."""
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
        _tls.s.headers["User-Agent"] = "Mozilla/5.0"
    last = None
    for _ in range(retries):
        try:
            r = _tls.s.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last = e
    raise RuntimeError(f"failed after {retries} tries: {url}") from last


def list_prefixes(prefix: str) -> list[str]:
    """List immediate child prefixes under a bucket prefix (paginated)."""
    out, marker = [], ""
    while True:
        url = f"{BUCKET}?delimiter=/&prefix={prefix}&marker={marker}"
        root = ElementTree.fromstring(_get(url))
        out += [p.find(f"{NS}Prefix").text for p in root.iter(f"{NS}CommonPrefixes")]
        if root.find(f"{NS}IsTruncated").text != "true":
            return out
        marker = out[-1]


def list_keys(prefix: str) -> list[str]:
    out, marker = [], ""
    while True:
        url = f"{BUCKET}?prefix={prefix}&marker={marker}"
        root = ElementTree.fromstring(_get(url))
        keys = [c.find(f"{NS}Key").text for c in root.iter(f"{NS}Contents")]
        out += [k for k in keys if k.endswith(".zip")]
        if root.find(f"{NS}IsTruncated").text != "true":
            return out
        marker = keys[-1]


def read_zip_csv(key: str) -> list[list[str]]:
    raw = _get(f"{BUCKET.replace('s3-ap-northeast-1.amazonaws.com/', '')}/{key}")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            rows = list(csv.reader(io.TextIOWrapper(f, "utf-8")))
    return [r for r in rows if r and not r[0][0].isalpha()]  # drop header if present


def _ts_to_utc(series: pd.Series) -> pd.Series:
    """Epoch ints of mixed precision (ms pre-2025 files, us after) -> datetime."""
    v = series.astype("int64")
    v = v.where(v < 10**14, v // 1000)  # us -> ms
    return pd.to_datetime(v, unit="ms", utc=True)


def fetch_symbol_klines(sym: str) -> pd.DataFrame | None:
    keys = list_keys(f"data/futures/um/monthly/klines/{sym}/1d/")
    rows: list[list[str]] = []
    for k in keys:
        rows += read_zip_csv(k)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = pd.to_numeric(df[c])
    df["n_trades"] = pd.to_numeric(df["n_trades"]).astype("int64")
    df["taker_buy_volume"] = pd.to_numeric(df["taker_buy_volume"])
    df["date"] = _ts_to_utc(df["open_time"]).dt.date
    df["symbol"] = sym
    return df[["symbol", "date", "open", "high", "low", "close",
               "volume", "quote_volume", "n_trades", "taker_buy_volume"]]


def fetch_symbol_funding(sym: str) -> pd.DataFrame | None:
    keys = list_keys(f"data/futures/um/monthly/fundingRate/{sym}/")
    rows: list[list[str]] = []
    for k in keys:
        rows += read_zip_csv(k)
    if not rows:
        return None
    df = pd.DataFrame(rows).iloc[:, :3]
    df.columns = ["calc_time", "funding_interval_hours", "funding_rate"]
    df["funding_time"] = _ts_to_utc(df["calc_time"])
    df["funding_rate"] = pd.to_numeric(df["funding_rate"])
    df["symbol"] = sym
    return df[["symbol", "funding_time", "funding_rate"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefixes = list_prefixes("data/futures/um/monthly/klines/")
    symbols = sorted(
        p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes
    )
    symbols = [s for s in symbols if s.endswith("USDT")]
    # Quarterly-dated contracts (e.g. BTCUSDT_210625) are not perpetuals.
    symbols = [s for s in symbols if not re.search(r"_\d{6}$", s)]
    print(f"{len(symbols)} USDT perp symbols", flush=True)

    jobs = (
        ("klines", fetch_symbol_klines, "perp_klines_1d.parquet"),
        ("funding", fetch_symbol_funding, "perp_funding.parquet"),
    )
    if len(sys.argv) > 1:
        jobs = tuple(j for j in jobs if j[0] in sys.argv[1:])
    for name, fn, out in jobs:
        parts, failed = [], []
        with ThreadPoolExecutor(N_WORKERS) as ex:
            futs = {ex.submit(fn, s): s for s in symbols}
            for i, fut in enumerate(as_completed(futs)):
                sym = futs[fut]
                try:
                    df = fut.result()
                    if df is not None:
                        parts.append(df)
                except Exception as e:  # noqa: BLE001
                    failed.append(sym)
                    print(f"FAIL {name} {sym}: {e}", flush=True)
                if (i + 1) % 50 == 0:
                    print(f"{name}: {i + 1}/{len(symbols)}", flush=True)
        full = pd.concat(parts, ignore_index=True)
        full.to_parquet(OUT_DIR / out, index=False)
        print(f"{name}: {len(full):,} rows, {full.symbol.nunique()} symbols, "
              f"{len(failed)} failed -> {out}", flush=True)
        if failed:
            print(f"failed {name} symbols: {failed}", flush=True)


if __name__ == "__main__":
    main()
