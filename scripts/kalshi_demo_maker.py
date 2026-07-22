"""FV-maker v0 on Kalshi DEMO — BTC hourly ladders, wide safe quotes.

The equilibrium test the replay sim cannot provide: real resting orders in a
real (demo) book, measuring fill rate, adverse selection, and whether other
makers undercut. Fake money; risk caps anyway (habits transfer to live).

Policy each cycle (10s):
  spot   Coinbase BTC-USD last trade (BRTI-constituent proxy; Kalshi settles
         on the 60s BRTI TWAP)
  sigma  EWMA of 1s log returns from the in-process spot buffer
  for each KXBTC hourly T-strike within +/-0.8% of spot, 10-50min to close:
    FV = Phi( ln(S/K) / (sigma*sqrt(tau)) )
    delta = PHI0/(sigma*sqrt(tau)) * 3*sigma*sqrt(CYCLE_S)   [q99~3sigma]
    quote post_only bid/ask at FV -/+ delta (sticky: requote only if the
    target moved > STICKY from the resting price), 10 contracts/side
  inventory cap 50/market -> quote reduce-only side; skip last 10min (TWAP
  gamma); cancel everything on exit or on daily loss > $25.

Logs: data/raw/polymarket/demomaker_*.jsonl.gz (sink format).

Usage: .venv/bin/python scripts/kalshi_demo_maker.py [--cycles N]
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
import urllib.request
import uuid
from collections import deque
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "kc", ROOT / "scripts" / "kalshi_client.py")
kc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kc)
import polymarket_recorder as rec  # noqa: E402  (Sink)

PHI0 = 0.3989
CYCLE_S = 10.0
STICKY = 0.02
SIZE = 10
INV_CAP = 50
DAILY_LOSS_STOP = 25.0
BAND = 0.05   # strike band vs spot; nearest N picked below
MAX_STRIKES = 12


def spot_now() -> float:
    req = urllib.request.Request(
        "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
        headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return float(json.load(r)["price"])


class Vol:
    def __init__(self, halflife_s: float = 300.0):
        self.hl = halflife_s
        self.var = None
        self.last = None
        self.t = None

    def update(self, px: float, t: float) -> None:
        if self.last is not None and t > self.t:
            dt = t - self.t
            r2 = math.log(px / self.last) ** 2 / dt  # per-second variance
            a = 1 - 0.5 ** (dt / self.hl)
            self.var = r2 if self.var is None else (1 - a) * self.var + a * r2
        self.last, self.t = px, t

    @property
    def sigma_1s(self) -> float:
        return math.sqrt(self.var) if self.var else 3e-4


def main() -> None:
    max_cycles = None
    if "--cycles" in sys.argv:
        max_cycles = int(sys.argv[sys.argv.index("--cycles") + 1])
    k = kc.Kalshi("demo")
    sink = rec.Sink("demomaker")
    vol = Vol()
    my_orders: dict[str, dict] = {}  # order_id -> {ticker, side, price}
    start_bal = float(k.get("/portfolio/balance")["balance_dollars"])
    print(f"demo maker start, balance ${start_bal}", flush=True)

    n = 0
    while max_cycles is None or n < max_cycles:
        t0 = time.time()
        try:
            S = spot_now()
            vol.update(S, t0)
            sig = vol.sigma_1s

            bal = float(k.get("/portfolio/balance")["balance_dollars"])
            if bal < start_bal - DAILY_LOSS_STOP:
                for oid in list(my_orders):
                    try:
                        k.delete(f"/portfolio/events/orders/{oid}")
                    except Exception:  # noqa: BLE001
                        pass
                sink.write("maker_stop", {"balance": bal})
                print("loss stop hit, all orders canceled", flush=True)
                break

            pos = {p["ticker"]: float(p.get("position_fp") or p.get("position", 0))
                   for p in k.get("/portfolio/positions").get(
                       "market_positions", [])}

            mkts = []
            for series in ("KXBTC", "KXBTCD"):
                mkts += k.get(f"/markets?series_ticker={series}"
                              "&status=open&limit=200")["markets"]
            now = time.time()
            cands = []
            for m in mkts:
                fs = m.get("floor_strike")
                if not fs or abs(fs / S - 1) > BAND:
                    continue
                close = time.mktime(time.strptime(
                    m["close_time"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
                tau = close - now
                if not 600 < tau < 16 * 3600:
                    continue
                st = sig * math.sqrt(tau)
                # T market = P(S>floor); B market = P(floor<S<cap)
                fv = norm.cdf(math.log(S / fs) / st)
                cap = m.get("cap_strike")
                if cap:
                    fv -= norm.cdf(math.log(S / cap) / st)
                if not 0.10 < fv < 0.90:
                    continue
                cands.append((abs(fv - 0.5), m, fv, tau))
            cands.sort(key=lambda x: x[0])
            targets = {}
            for _, m, fv, tau in cands[:MAX_STRIKES]:
                st = sig * math.sqrt(tau)
                delta = min(0.25, PHI0 / st * 3 * sig * math.sqrt(CYCLE_S))
                inv = pos.get(m["ticker"], 0.0)
                # clamp to the live book: post-only join-or-behind, never cross
                try:
                    ob = k.get(f"/markets/{m['ticker']}/orderbook?depth=1"
                               ).get("orderbook_fp", {})
                    yb = ob.get("yes_dollars") or []
                    nb = ob.get("no_dollars") or []
                    book_bid = float(yb[-1][0]) if yb else None
                    book_ask = 1.0 - float(nb[-1][0]) if nb else None
                except Exception:  # noqa: BLE001
                    book_bid = book_ask = None
                q = {}
                if inv < INV_CAP:
                    b = max(0.01, round(fv - delta, 2))
                    if book_ask is not None:
                        b = min(b, round(book_ask - 0.01, 2))
                    if b >= 0.01:
                        q["bid"] = b
                if inv > -INV_CAP:
                    a = min(0.99, round(fv + delta, 2))
                    if book_bid is not None:
                        a = max(a, round(book_bid + 0.01, 2))
                    if a <= 0.99:
                        q["ask"] = a
                targets[m["ticker"]] = {"fv": fv, "delta": delta, **q}

            # reconcile: cancel stale/off-target orders, place missing ones
            open_orders = k.get("/portfolio/orders?status=resting"
                                ).get("orders", [])
            live = {}
            for o in open_orders:
                tick, oid = o["ticker"], o["order_id"]
                side = "bid" if o.get("side") == "yes" and o.get(
                    "action", "buy") == "buy" else "ask"
                # kalshi v1 listing: side yes/no + action; normalize via price
                px = float(o.get("yes_price_dollars") or 0) or None
                live[(tick, side)] = (oid, px)
            canceled_any = False
            for (tick, side), (oid, px) in live.items():
                tgt = targets.get(tick, {}).get(side)
                if tgt is None or px is None or abs(px - tgt) > STICKY:
                    try:
                        k.delete(f"/portfolio/events/orders/{oid}")
                        canceled_any = True
                    except Exception:  # noqa: BLE001
                        pass
                    live[(tick, side)] = None
            if canceled_any:
                time.sleep(1.0)  # let cancels land before re-quoting: avoids
                # post-only crossing our own in-flight canceled orders
            for tick, tgt in targets.items():
                for side in ("bid", "ask"):
                    if side not in tgt or live.get((tick, side)):
                        continue
                    # v2 API: side bid=buy yes, ask=sell yes; price is yes price
                    price = tgt[side]
                    try:
                        k.post("/portfolio/events/orders", {
                            "ticker": tick,
                            "client_order_id": str(uuid.uuid4()),
                            "side": side, "count": f"{SIZE}.00",
                            "price": f"{price:.4f}",
                            "time_in_force": "good_till_canceled",
                            "self_trade_prevention_type": "maker",
                            "post_only": True,
                            "cancel_order_on_pause": True})
                    except Exception as e:  # noqa: BLE001
                        sink.write("order_err", {"ticker": tick, "side": side,
                                                 "err": str(e)[:200]})
            sink.write("maker_cycle", {
                "spot": S, "sigma_1s": sig, "balance": bal,
                "targets": {t: {kk: round(v, 4) if isinstance(v, float) else v
                                for kk, v in d.items()}
                            for t, d in targets.items()},
                "n_open": len(open_orders), "pos": pos})
        except Exception as e:  # noqa: BLE001
            sink.write("maker_err", str(e)[:300])
        sink.flush()
        n += 1
        time.sleep(max(0.0, CYCLE_S - (time.time() - t0)))
    print("maker loop ended", flush=True)


if __name__ == "__main__":
    main()
