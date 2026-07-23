"""Poll Kalshi BTC market orderbooks alongside the Polymarket recorder.

Public API, no account. Every POLL_SEC, fetch orderbooks for open KXBTC
(hourly) and KXBTCD (daily) markets closing within HORIZON; market list
refreshes every LIST_SEC. Output shares the Polymarket recorder's directory
and format: JSONL {"t_local", "src": "kalshi", "msg"} gzip, hourly files.

Usage:
  .venv/bin/python scripts/kalshi_poller.py            # run until killed
  .venv/bin/python scripts/kalshi_poller.py --probe 30
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone

from polymarket_recorder import OUT, Sink  # noqa: F401  (shared sink/dir)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = ("KXBTC", "KXBTCD")
POLL_SEC = 10
LIST_SEC = 60
HORIZON_SEC = 2 * 3600


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE}{path}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


_ATM_CACHE: dict[str, bool] = {}  # ticker -> last real-orderbook ATM verdict


def list_markets() -> list[dict]:
    out = []
    now = time.time()
    for s in SERIES:
        d = _get(f"/markets?series_ticker={s}&status=open&limit=200")
        for mk in d.get("markets", []):
            close = datetime.strptime(
                mk["close_time"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc).timestamp()
            if close - now >= HORIZON_SEC:
                continue
            # full ladder recorded; near-the-money flag sets the poll tier
            # (ATM books every cycle, far strikes every 6th) so dutching /
            # sum-of-ladder consistency is answerable without 24 req/s
            yes_bid = 1.0 - float(mk.get("no_ask_dollars") or 1.0)
            yes_ask = 1.0 - float(mk.get("no_bid_dollars") or 0.0)
            # ATM tier from the LAST REAL orderbook (listing bid/ask lies
            # on one-sided books); unseen tickers default to ATM-frequency
            out.append({"ticker": mk["ticker"],
                        "close_time": mk["close_time"],
                        "floor_strike": mk.get("floor_strike"),
                        "cap_strike": mk.get("cap_strike"),
                        "yes_bid": yes_bid, "yes_ask": yes_ask,
                        "atm": _ATM_CACHE.get(mk["ticker"], True)})
    return out


def main() -> None:
    stop = None
    if "--probe" in sys.argv:
        stop = time.time() + float(sys.argv[sys.argv.index("--probe") + 1])
    sink = Sink("kalshi")
    markets: list[dict] = []
    last_list = 0.0
    n = 0
    while stop is None or time.time() < stop:
        t0 = time.time()
        try:
            if t0 - last_list > LIST_SEC:
                markets = list_markets()
                sink.write("kalshi_markets", markets)
                last_list = t0
            cycle = n
            for mk in markets:
                if not mk.get("atm", True) and cycle % 6 != 0:
                    continue
                ob = _get(f"/markets/{mk['ticker']}/orderbook")
                sink.write("kalshi", {"ticker": mk["ticker"], "ob": ob})
                obf = ob.get("orderbook_fp", {}) if isinstance(ob, dict) else {}
                yb = obf.get("yes_dollars") or []
                nb = obf.get("no_dollars") or []
                two_sided = bool(yb) and bool(nb)
                if two_sided:
                    mid = (float(yb[-1][0]) + 1.0 - float(nb[-1][0])) / 2
                    _ATM_CACHE[mk["ticker"]] = 0.03 <= mid <= 0.97
                else:
                    _ATM_CACHE[mk["ticker"]] = False
            n += 1
        except Exception as e:  # noqa: BLE001
            sink.write("kalshi_err", str(e))
        sink.flush()
        time.sleep(max(0.0, POLL_SEC - (time.time() - t0)))
    if sink.fh:
        sink.fh.close()
    if stop:
        print(f"probe: {n} poll cycles, {len(markets)} markets tracked "
              f"({sum(1 for m in markets if m.get('atm'))} ATM)", flush=True)


if __name__ == "__main__":
    main()
