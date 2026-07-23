"""Record Polymarket up/down CLOB books + Binance spot trades for the
latency-window study (Polymarket venue kill-test #1).

Streams, all public, no account:
  - Polymarket CLOB market channel for active {btc,eth,xrp}-updown-{5m,15m,1h}
    markets (book snapshots, price changes, trades). Markets rotate every few
    minutes; a discovery loop re-resolves the active set every 30s and the WS
    connection is rebuilt when the set changes.
  - Binance spot trades for BTC/ETH/XRP-USDT via data-stream.binance.vision.

Every message is stored raw as one JSONL line {"t_local": <unix us>, "src":
..., "msg": ...}, gzip files rotated hourly under data/raw/polymarket/.
Local receive timestamps on both feeds are what the latency analysis needs.

Usage:
  .venv/bin/python scripts/polymarket_recorder.py           # run until killed
  .venv/bin/python scripts/polymarket_recorder.py --probe 25  # smoke test
"""

from __future__ import annotations

import asyncio
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

import websockets

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket"
GAMMA = "https://gamma-api.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
BINANCE_WS = ("wss://data-stream.binance.vision/stream?streams="
              "btcusdt@trade/ethusdt@trade/xrpusdt@trade")
DISCOVER_SEC = 30
SERIES_KEY = "btc-updown"  # BTC only: full 3-coin recording is ~11 GB/day


class Sink:
    def __init__(self, name: str):
        self.name, self.fh, self.hour = name, None, None

    def write(self, src: str, msg) -> None:
        now = time.time()
        hour = int(now // 3600)
        if hour != self.hour:
            if self.fh:
                self.fh.close()
            OUT.mkdir(parents=True, exist_ok=True)
            path = OUT / f"{self.name}_{time.strftime('%Y%m%d_%H', time.gmtime(now))}.jsonl.gz"
            self.fh = gzip.open(path, "at")
            self.hour = hour
        self.fh.write(json.dumps(
            {"t_local": int(now * 1e6), "src": src, "msg": msg},
            separators=(",", ":")) + "\n")

    def flush(self) -> None:
        if self.fh:
            self.fh.flush()


def discover() -> dict[str, dict]:
    """Active updown markets -> {token_id: {slug, end}}."""
    url = (f"{GAMMA}/events?closed=false&limit=100&tag_slug=up-or-down"
           f"&order=endDate&ascending=true")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        events = json.load(r)
    toks: dict[str, dict] = {}
    import calendar
    now_ts = time.time()
    horizon = now_ts + 45 * 60
    for e in events:
        if SERIES_KEY not in (e.get("slug") or ""):
            continue
        end = e.get("endDate")
        if end:
            end_ts = calendar.timegm(time.strptime(end, "%Y-%m-%dT%H:%M:%SZ"))
            # skip stale zombies (closed=false but long past) AND far-future:
            # zombies churned the set 33x/hour -> reconnect data loss
            if end_ts > horizon or end_ts < now_ts - 120:
                continue
        for mk in e.get("markets", []):
            ids = mk.get("clobTokenIds")
            if isinstance(ids, str):
                ids = json.loads(ids)
            for tid in ids or []:
                toks[tid] = {"slug": e["slug"], "end": e.get("endDate")}
    return toks


async def poly_loop(sink: Sink, stop: float | None) -> None:
    """Reconnect only when the active market set changes (~every 5 min),
    so full book snapshots are not re-sent every discovery poll."""
    ws, assets = None, frozenset()
    while stop is None or time.time() < stop:
        try:
            toks = discover()
            new = frozenset(toks)
            if new and new != assets:
                sink.write("discovery", {tid: v for tid, v in toks.items()})
                if ws is not None:
                    await ws.close()
                ws = await websockets.connect(CLOB_WS, ping_interval=10)
                await ws.send(json.dumps(
                    {"assets_ids": sorted(new), "type": "market"}))
                assets = new
            if ws is None:
                await asyncio.sleep(DISCOVER_SEC)
                continue
            deadline = time.time() + DISCOVER_SEC
            while time.time() < deadline and (stop is None or time.time() < stop):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    sink.write("poly", json.loads(raw))
                except asyncio.TimeoutError:
                    pass
        except Exception as e:  # noqa: BLE001
            sink.write("poly_err", str(e))
            if ws is not None:
                try:
                    await ws.close()
                except Exception:  # noqa: BLE001
                    pass
            ws, assets = None, frozenset()
            await asyncio.sleep(3)
        sink.flush()
    if ws is not None:
        await ws.close()


async def binance_loop(sink: Sink, stop: float | None) -> None:
    while stop is None or time.time() < stop:
        try:
            async with websockets.connect(BINANCE_WS, ping_interval=15) as ws:
                while stop is None or time.time() < stop:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    sink.write("binance", json.loads(raw))
        except Exception as e:  # noqa: BLE001
            sink.write("binance_err", str(e))
            await asyncio.sleep(3)
        sink.flush()


async def main() -> None:
    probe = None
    if "--probe" in sys.argv:
        probe = time.time() + float(sys.argv[sys.argv.index("--probe") + 1])
    poly_sink, bin_sink = Sink("poly"), Sink("binance")
    await asyncio.gather(poly_loop(poly_sink, probe),
                         binance_loop(bin_sink, probe))
    for s in (poly_sink, bin_sink):
        if s.fh:
            s.fh.close()
    if probe:
        import collections
        counts = collections.Counter()
        for f in sorted(OUT.glob("*.jsonl.gz")):
            try:
                for line in gzip.open(f, "rt"):
                    counts[json.loads(line)["src"]] += 1
            except EOFError:
                pass
        print("probe message counts:", dict(counts), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
