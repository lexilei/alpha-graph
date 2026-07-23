"""Minimal Coinbase Advanced Trade client (CDP key, Ed25519 JWT auth).

Credentials live outside the repo in data/private/:
  coinbase_key_name.txt   organizations/{org}/apiKeys/{key}
  coinbase_secret.txt     base64 Ed25519 (64 bytes: seed + pubkey)

Run as a script: verify key permissions, then list perpetual products with
their view/trade eligibility flags and the account fee tier.

Usage: .venv/bin/python scripts/coinbase_client.py
"""

from __future__ import annotations

import base64
import json
import secrets
import time
import urllib.request
from pathlib import Path

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

PRIVATE = Path(__file__).resolve().parents[1] / "data" / "private"
HOST = "api.coinbase.com"


class Coinbase:
    def __init__(self):
        self.name = (PRIVATE / "coinbase_key_name.txt").read_text().strip()
        raw = base64.b64decode((PRIVATE / "coinbase_secret.txt").read_text().strip())
        self.key = Ed25519PrivateKey.from_private_bytes(raw[:32])

    def _jwt(self, method: str, path: str) -> str:
        uri = f"{method} {HOST}{path.split('?')[0]}"
        now = int(time.time())
        return jwt.encode(
            {"sub": self.name, "iss": "cdp", "nbf": now, "exp": now + 120,
             "uri": uri},
            self.key, algorithm="EdDSA",
            headers={"kid": self.name, "nonce": secrets.token_hex(16)})

    def get(self, path: str):
        req = urllib.request.Request(
            f"https://{HOST}{path}",
            headers={"Authorization": f"Bearer {self._jwt('GET', path)}",
                     "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GET {path} -> {e.code}: "
                               f"{e.read().decode()[:300]}") from e


def main() -> None:
    c = Coinbase()
    perms = c.get("/api/v3/brokerage/key_permissions")
    print("key permissions:", perms)

    prods = c.get("/api/v3/brokerage/products?product_type=FUTURE"
                  "&contract_expiry_type=PERPETUAL").get("products", [])
    print(f"\nperpetual products visible: {len(prods)}")
    tradable = [p for p in prods if not p.get("trading_disabled")
                and not p.get("view_only")]
    print(f"tradable (not disabled/view-only): {len(tradable)}")
    for p in tradable[:10]:
        print(f"  {p['product_id']:22s} status={p.get('status')} "
              f"disabled={p.get('trading_disabled')} view={p.get('view_only')}")

    try:
        fees = c.get("/api/v3/brokerage/transaction_summary")
        ft = fees.get("fee_tier", {})
        print(f"\nfee tier: {ft.get('pricing_tier')} maker "
              f"{ft.get('maker_fee_rate')} taker {ft.get('taker_fee_rate')}")
    except Exception as e:  # noqa: BLE001
        print("fee summary:", str(e)[:150])


if __name__ == "__main__":
    main()
