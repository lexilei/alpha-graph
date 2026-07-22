"""Minimal Kalshi trading client (demo or production).

Auth per docs.kalshi.com: each request carries
  KALSHI-ACCESS-KEY        key id
  KALSHI-ACCESS-TIMESTAMP  unix ms
  KALSHI-ACCESS-SIGNATURE  base64 RSA-PSS-SHA256 over ts + METHOD + path
where path excludes query parameters. Credentials live OUTSIDE the repo in
data/private/ (kalshi_demo_key.pem + kalshi_demo_keyid.txt; production uses
kalshi_key.pem + kalshi_keyid.txt).

Run as a script for a demo smoke test: balance -> pick an open KXBTC market
-> rest a 1-cent bid -> verify -> cancel.

Usage: .venv/bin/python scripts/kalshi_client.py
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
import uuid
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

PRIVATE = Path(__file__).resolve().parents[1] / "data" / "private"


class Kalshi:
    def __init__(self, env: str = "demo"):
        host = "demo-api.kalshi.co" if env == "demo" else "api.elections.kalshi.com"
        self.base = f"https://{host}/trade-api/v2"
        suffix = "_demo" if env == "demo" else ""
        self.key_id = (PRIVATE / f"kalshi{suffix}_keyid.txt").read_text().strip()
        self.key = serialization.load_pem_private_key(
            (PRIVATE / f"kalshi{suffix}_key.pem").read_bytes(), password=None)

    def _headers(self, method: str, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method}/trade-api/v2{path.split('?')[0]}".encode()
        sig = self.key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self.key_id,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"}

    def request(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=self._headers(method, path), method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> {e.code}: "
                               f"{e.read().decode()[:300]}") from e

    def get(self, path: str):
        return self.request("GET", path)

    def post(self, path: str, body: dict):
        return self.request("POST", path, body)

    def delete(self, path: str):
        return self.request("DELETE", path)


def smoke_test() -> None:
    k = Kalshi("demo")
    bal = k.get("/portfolio/balance")
    print("balance:", bal)

    mkts = k.get("/markets?series_ticker=KXBTC&status=open&limit=20")["markets"]
    m = next((x for x in mkts
              if 0.05 < float(x["yes_ask_dollars"] or 0) < 0.95), mkts[0])
    print(f"market: {m['ticker']} yes {m['yes_bid_dollars']}/{m['yes_ask_dollars']}")

    order = k.post("/portfolio/events/orders", {
        "ticker": m["ticker"],
        "client_order_id": str(uuid.uuid4()),
        "side": "bid",                       # buy YES
        "count": "1.00", "price": "0.0100",  # 1-cent bid, rests far away
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "maker",
        "post_only": True,
    })
    o = order.get("order", order)
    oid = o.get("order_id") or o.get("id")
    print("placed:", oid, o.get("status"))

    # listing still lives at the v1 path; create/cancel are v2 event paths
    resting = k.get("/portfolio/orders?status=resting").get("orders", [])
    print("resting orders:", [(x.get("ticker"), x.get("status")) for x in resting])

    out = k.delete(f"/portfolio/events/orders/{oid}")
    print("cancel response:", str(out)[:200])


if __name__ == "__main__":
    smoke_test()
