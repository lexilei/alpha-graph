"""Production go-live smoke test (panel review, unanimous condition).

Every safety control was verified on DEMO only; this exercises the PROD
endpoint/key once, with ~2 cents at risk, before the maker quotes a dollar.

Phases:
  read      (default) read-only: balance, schema (balance_dollars +
            cents-portfolio_value), no resting orphans, market pagination,
            orderbook shape on a live ladder.
  arm       everything above PLUS the two-order live test:
              1. place 1 lot far-OTM @ 1c post-only w/ expiration now+120
                 -> verify RESTING at t+45s -> verify GONE at t+140s
                 (prod dead-man proof: the control the whole unattended
                 safety case rests on)
              2. place a second 1c order -> DELETE it -> verify cancel works
            Requires data/private/maker_env.txt == prod is NOT required —
            this script talks to prod explicitly and independently.

Usage: .venv/bin/python scripts/prod_smoke.py [arm]
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import calendar
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kalshi_client as kc

PASS, FAIL = "PASS", "FAIL"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures += 1


def main() -> None:
    arm = "arm" in sys.argv
    k = kc.Kalshi("prod")

    b = k.get("/portfolio/balance")
    bal = b.get("balance_dollars")
    check("balance readable", bal is not None, f"balance_dollars={bal}")
    check("balance schema (cents portfolio_value present)",
          "portfolio_value" in b, f"portfolio_value={b.get('portfolio_value')}")
    check("balance ~= expected bankroll", bal is not None
          and 40.0 <= float(bal) <= 60.0, f"{bal} (expected ~49.5)")

    resting = k.get("/portfolio/orders?status=resting").get("orders", [])
    check("no orphan resting orders", len(resting) == 0, f"n={len(resting)}")

    mkts, cursor = [], ""
    for _ in range(12):
        r = k.get("/markets?series_ticker=KXBTCD&status=open&limit=200"
                  + (f"&cursor={cursor}" if cursor else ""))
        mkts += r.get("markets", [])
        cursor = r.get("cursor") or ""
        if not cursor:
            break
    check("market pagination", len(mkts) > 200, f"{len(mkts)} mkts")
    now = time.time()

    def tau(m):
        return calendar.timegm(time.strptime(
            m["close_time"], "%Y-%m-%dT%H:%M:%SZ")) - now

    live = [m for m in mkts if m.get("floor_strike") and 3600 < tau(m) < 40 * 3600]
    check("live-window markets exist", len(live) > 10, f"{len(live)}")
    probe = max(live, key=lambda x: x["floor_strike"])  # deepest OTM
    ob = k.get(f"/markets/{probe['ticker']}/orderbook?depth=1")
    check("orderbook schema (orderbook_fp)", "orderbook_fp" in ob,
          str(list(ob.keys())))
    fl = k.get("/portfolio/fills?limit=10")
    check("fills endpoint readable", "fills" in fl, f"n={len(fl.get('fills', []))}")

    if not arm:
        print(f"\nread-only phase done, failures={failures}. "
              "Run with 'arm' for the live order test.")
        sys.exit(1 if failures else 0)

    # ---- live order test: ~2 cents at risk, deepest-OTM strike ----
    print(f"\nARM: live order test on {probe['ticker']} "
          f"(floor {probe['floor_strike']})")
    body = {"ticker": probe["ticker"], "client_order_id": str(uuid.uuid4()),
            "side": "bid", "count": "1.00", "price": "0.0100",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "maker", "post_only": True,
            "expiration_time": int(time.time()) + 120}
    k.post("/portfolio/events/orders", body)

    def mine():
        os_ = k.get("/portfolio/orders?status=resting").get("orders", [])
        return [o for o in os_ if o["ticker"] == probe["ticker"]]

    time.sleep(45)
    check("dead-man order RESTING at t+45s", len(mine()) == 1)
    time.sleep(95)
    gone = len(mine()) == 0
    check("dead-man order GONE at t+140s (PROD self-expiry)", gone)
    if not gone:
        for o in mine():
            k.delete(f"/portfolio/events/orders/{o['order_id']}")
        print("  cleaned up manually — DEAD-MAN NOT PROVEN ON PROD")

    body2 = dict(body, client_order_id=str(uuid.uuid4()),
                 expiration_time=int(time.time()) + 300)
    r2 = k.post("/portfolio/events/orders", body2)
    oid = (r2.get("order") or {}).get("order_id") or r2.get("order_id")
    time.sleep(3)
    k.delete(f"/portfolio/events/orders/{oid}")
    time.sleep(3)
    check("explicit cancel works", len(mine()) == 0)

    print(f"\nsmoke complete, failures={failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
