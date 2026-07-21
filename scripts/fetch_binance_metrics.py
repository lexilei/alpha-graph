"""Download Binance UM futures daily metrics (OI, long-short ratios).

Source: data.binance.vision, data/futures/um/daily/metrics/{SYMBOL}/ —
one zip per symbol per day containing 5-minute snapshots of:
  sum_open_interest, sum_open_interest_value,
  count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
  count_long_short_ratio, sum_taker_long_short_vol_ratio

To keep the request count sane (~250k vs 1.3M) the universe is restricted
to symbols that ever ranked in the top 150 by 30d median quote volume in
the existing klines panel. Each numeric column is aggregated to daily
last-snapshot and daily mean.

Output: data/raw/binance/perp_metrics_daily.parquet

Usage: .venv/bin/python scripts/fetch_binance_metrics.py
"""

from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from fetch_binance_perp import OUT_DIR, _get, list_keys

RANK_CUTOFF = 150
N_WORKERS = 24

NUM_COLS = ["sum_open_interest", "sum_open_interest_value",
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


def eligible_symbols() -> list[str]:
    k = pd.read_parquet(OUT_DIR / "perp_klines_1d.parquet",
                        columns=["symbol", "date", "quote_volume", "close"])
    k["date"] = pd.to_datetime(k["date"])
    qv = k.pivot(index="date", columns="symbol", values="quote_volume")
    close = k.pivot(index="date", columns="symbol", values="close")
    med = qv.rolling(30, min_periods=30).median()
    rank = med.where(close.notna()).rank(axis=1, ascending=False)
    keep = rank.min().le(RANK_CUTOFF)
    return sorted(keep[keep].index)


def fetch_one(key: str) -> pd.DataFrame | None:
    raw = _get(f"https://data.binance.vision/{key}")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    if df.empty:
        return None
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    ts = pd.to_datetime(df["create_time"], format="mixed")
    day = ts.dt.normalize().iloc[-1]
    last = df[NUM_COLS].iloc[-1]
    mean = df[NUM_COLS].mean()
    row = {"symbol": df["symbol"].iloc[0], "date": day}
    row |= {f"{c}_last": last[c] for c in NUM_COLS}
    row |= {f"{c}_mean": mean[c] for c in NUM_COLS}
    return pd.DataFrame([row])


def fetch_symbol(sym: str) -> pd.DataFrame | None:
    keys = list_keys(f"data/futures/um/daily/metrics/{sym}/")
    parts = []
    for k in keys:
        try:
            df = fetch_one(k)
            if df is not None:
                parts.append(df)
        except Exception as e:  # noqa: BLE001 - one bad day shouldn't kill a symbol
            print(f"skip {k}: {e}", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else None


def main() -> None:
    symbols = eligible_symbols()
    print(f"{len(symbols)} symbols ever in top {RANK_CUTOFF}", flush=True)
    parts, failed = [], []
    with ThreadPoolExecutor(N_WORKERS) as ex:
        futs = {ex.submit(fetch_symbol, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futs)):
            sym = futs[fut]
            try:
                df = fut.result()
                if df is not None:
                    parts.append(df)
            except Exception as e:  # noqa: BLE001
                failed.append(sym)
                print(f"FAIL {sym}: {e}", flush=True)
            if (i + 1) % 10 == 0:
                print(f"metrics: {i + 1}/{len(symbols)} symbols", flush=True)
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(OUT_DIR / "perp_metrics_daily.parquet", index=False)
    print(f"metrics: {len(full):,} rows, {full.symbol.nunique()} symbols, "
          f"{len(failed)} failed -> perp_metrics_daily.parquet", flush=True)
    if failed:
        print(f"failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
