"""Record the Deribit BTC option surface for the Kalshi-bracket study.

Every POLL_SEC: one public call returns bid/ask/mark_iv for all live BTC
options (get_book_summary_by_currency) plus the index price. The risk-neutral
distribution rebuilt from this surface is the FV reference for Kalshi's
daily/weekly BTC bracket ladders (vol_smile SVI/BKM machinery downstream).

Output: data/raw/deribit/deribit_*.jsonl.gz (recorder sink format).

Usage: .venv/bin/python scripts/deribit_recorder.py [--probe SECONDS]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

import polymarket_recorder as rec

POLL_SEC = 60
BASE = "https://www.deribit.com/api/v2/public"
OUT = rec.OUT.parent / "deribit"


class DSink(rec.Sink):
    def write(self, src, msg):
        rec.OUT = OUT
        super().write(src, msg)


def _get(path: str):
    req = urllib.request.Request(f"{BASE}{path}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)["result"]


def main() -> None:
    stop = None
    if "--probe" in sys.argv:
        stop = time.time() + float(sys.argv[sys.argv.index("--probe") + 1])
    sink = DSink("deribit")
    n = 0
    while stop is None or time.time() < stop:
        t0 = time.time()
        try:
            summ = _get("/get_book_summary_by_currency?currency=BTC&kind=option")
            idx = _get("/get_index_price?index_name=btc_usd")
            slim = [{k: s.get(k) for k in
                     ("instrument_name", "bid_price", "ask_price", "mark_price",
                      "mark_iv", "underlying_price", "open_interest", "volume")}
                    for s in summ]
            sink.write("deribit_surface", {"index": idx, "options": slim})
            n += 1
        except Exception as e:  # noqa: BLE001
            sink.write("deribit_err", str(e))
        sink.flush()
        time.sleep(max(0.0, POLL_SEC - (time.time() - t0)))
    if sink.fh:
        sink.fh.close()
    if stop:
        print(f"probe: {n} surface snapshots", flush=True)


if __name__ == "__main__":
    main()
