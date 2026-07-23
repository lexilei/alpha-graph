"""Weather category kill-test: are Kalshi daily-high markets still soft?

For settled daily-high-temperature markets across cities, take the market
price at fixed lead times before close (via public hourly candlesticks) and
compare with outcomes: calibration by price bucket, Brier score, and the
net-of-fee P&L of mechanically fading extreme buckets. No forecast model is
used — this measures whether the MARKET is sharp, i.e. whether building the
NOAA/HRRR pipeline (TODO #11) could pay.

Usage: .venv/bin/python scripts/weather_calibration.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CITIES = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHDEN",
          "KXHIGHLAX", "KXHIGHPHIL"]
LEADS_H = (18, 8)
MAX_PER_CITY = 200


def kget(path: str):
    req = urllib.request.Request(
        f"https://api.elections.kalshi.com/trade-api/v2{path}",
        headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def main() -> None:
    rows = []
    for city in CITIES:
        try:
            ms = kget(f"/markets?series_ticker={city}&status=settled"
                      f"&limit={MAX_PER_CITY}").get("markets", [])
        except Exception as e:  # noqa: BLE001
            print(f"{city}: series error {str(e)[:80]}")
            continue
        n_city = 0
        for m in ms:
            res = m.get("result")
            if res not in ("yes", "no"):
                continue
            close = int(datetime.strptime(
                m["close_time"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc).timestamp())
            try:
                cs = kget(f"/series/{city}/markets/{m['ticker']}/candlesticks"
                          f"?period_interval=60&start_ts={close - 30*3600}"
                          f"&end_ts={close}").get("candlesticks", [])
            except Exception:  # noqa: BLE001
                continue
            if not cs:
                continue
            n_city += 1
            for lead in LEADS_H:
                cutoff = close - lead * 3600
                prior = [c for c in cs if int(c.get("end_period_ts", 0)) <= cutoff]
                if not prior:
                    continue
                c = prior[-1]
                try:
                    yb = float(c["yes_bid"]["close_dollars"])
                    ya = float(c["yes_ask"]["close_dollars"])
                except (KeyError, TypeError, ValueError):
                    continue
                if ya - yb > 0.20:  # junk-wide book, mid meaningless
                    continue
                px = (yb + ya) / 2
                if not 0.005 <= px <= 0.995:
                    continue
                rows.append({"city": city, "ticker": m["ticker"],
                             "lead": lead, "p": px,
                             "y": 1.0 if res == "yes" else 0.0})
        print(f"{city}: {n_city} settled markets with candles", flush=True)
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no data")
        return
    print(f"\ntotal observations: {len(df)} across {df.ticker.nunique()} markets")
    for lead in LEADS_H:
        g = df[df.lead == lead]
        brier = ((g.p - g.y) ** 2).mean()
        base = g.y.mean()
        brier_clim = (base * (1 - base))
        print(f"\n== lead {lead}h: n={len(g)}, Brier {brier:.4f} "
              f"(climatology-ish bound {brier_clim:.4f})")
        g = g.assign(bucket=(g.p * 10).astype(int).clip(0, 9))
        cal = g.groupby("bucket").agg(n=("y", "size"), implied=("p", "mean"),
                                      hit=("y", "mean"))
        cal["gap"] = cal.hit - cal.implied
        print(cal.round(3).to_string())
        # mechanical fade of tails: sell yes above 90c? buy under 10c? net fee
        for lo, hi, act in ((0.0, 0.10, "BUY cheap tail"),
                            (0.90, 1.0, "SELL rich tail")):
            t = g[(g.p >= lo) & (g.p < hi)]
            if len(t) < 10:
                continue
            fee = 0.07 * t.p * (1 - t.p)
            pnl = (t.y - t.p - fee) if act.startswith("BUY") else (t.p - t.y - fee)
            print(f"  {act}: n={len(t)} mean pnl/contract {pnl.mean():+.4f}")


if __name__ == "__main__":
    main()
