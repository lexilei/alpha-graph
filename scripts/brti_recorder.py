"""Record BRTI-constituent spot trades: Coinbase + Kraken BTC-USD.

Kalshi BTC markets settle on the 60s average of CF Benchmarks' BRTI, whose
constituents are US-facing exchanges (Coinbase, Kraken, Bitstamp, Gemini,
LMAX) — NOT Binance. This records the two largest constituents' trade feeds
as the BRTI proxy for Kalshi fair value and the settlement-TWAP study.

Output: data/raw/polymarket/brti_*.jsonl.gz (same sink format).

Usage: .venv/bin/python scripts/brti_recorder.py [--probe SECONDS]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import websockets

from polymarket_recorder import Sink

COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
KRAKEN_WS = "wss://ws.kraken.com"
FLUSH_SEC = 5.0


def _tick_flush(sink: Sink, last: list[float]) -> None:
    # the hot recv loop never reaches the outer flush; without this, data
    # sits invisible in the gzip buffer and dies with the process on SIGKILL
    now = time.time()
    if now - last[0] > FLUSH_SEC:
        sink.flush()
        last[0] = now


async def coinbase_loop(sink: Sink, stop: float | None) -> None:
    sub = {"type": "subscribe", "product_ids": ["BTC-USD", "ETH-USD"],
           "channels": ["matches"]}
    last_flush = [0.0]
    while stop is None or time.time() < stop:
        try:
            async with websockets.connect(COINBASE_WS, ping_interval=15) as ws:
                await ws.send(json.dumps(sub))
                while stop is None or time.time() < stop:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                    if m.get("type") == "match":
                        sink.write("coinbase", {"p": m["price"], "s": m["size"],
                                                "t": m["time"],
                                                "prod": m.get("product_id")})
                        _tick_flush(sink, last_flush)
        except Exception as e:  # noqa: BLE001
            sink.write("coinbase_err", str(e))
            await asyncio.sleep(3)
        sink.flush()


async def kraken_loop(sink: Sink, stop: float | None) -> None:
    sub = {"event": "subscribe", "pair": ["XBT/USD", "ETH/USD"],
           "subscription": {"name": "trade"}}
    last_flush = [0.0]
    while stop is None or time.time() < stop:
        try:
            async with websockets.connect(KRAKEN_WS, ping_interval=15) as ws:
                await ws.send(json.dumps(sub))
                while stop is None or time.time() < stop:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                    if isinstance(m, list) and len(m) > 2 and m[2] == "trade":
                        sink.write("kraken", {"pair": m[-1], "trades": m[1]})
                        _tick_flush(sink, last_flush)
        except Exception as e:  # noqa: BLE001
            sink.write("kraken_err", str(e))
            await asyncio.sleep(3)
        sink.flush()


async def main() -> None:
    stop = None
    if "--probe" in sys.argv:
        stop = time.time() + float(sys.argv[sys.argv.index("--probe") + 1])
    sink = Sink("brti")
    await asyncio.gather(coinbase_loop(sink, stop), kraken_loop(sink, stop))
    if sink.fh:
        sink.fh.close()


if __name__ == "__main__":
    asyncio.run(main())
