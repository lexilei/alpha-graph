"""Record sportsbook odds (the-odds-api) vs Polymarket MLB game markets.

Reproduction study of the HN sportsbook-arb approach, MLB domain (the free
odds API has no esports). Every POLL_MIN minutes:
  - the-odds-api baseball_mlb h2h odds, eu region only (Pinnacle included);
    costs 1 credit/call, free tier 500/month. Calls stop when the remaining
    quota drops below RESERVE so the key is never exhausted.
  - Polymarket per-game MLB events (slug mlb-away-home-YYYY-MM-DD) via the
    free gamma API, plus full CLOB book depth for their tokens.

API key: data/raw/odds/apikey.txt (gitignored). Output JSONL.gz sinks in
data/raw/odds/, same format as the other recorders.

Usage:
  .venv/bin/python scripts/odds_poller.py            # run until killed
  .venv/bin/python scripts/odds_poller.py --probe 1  # one cycle, then exit
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import polymarket_recorder as rec

ROOT = Path(__file__).resolve().parents[1]
ODDS_DIR = ROOT / "data" / "raw" / "odds"
KEY = (ODDS_DIR / "apikey.txt").read_text().strip()
POLL_MIN = 15
RESERVE = 60
GAME_RE = re.compile(r"^mlb-\w+-\w+-(\d{4}-\d{2}-\d{2})$")




def _get(url: str, headers: dict | None = None):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r), dict(r.headers)


def fetch_odds(sink: rec.Sink) -> int | None:
    url = (f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
           f"?regions=eu&markets=h2h&oddsFormat=decimal&apiKey={KEY}")
    data, hdrs = _get(url)
    remaining = int(float(hdrs.get("x-requests-remaining", "0")))
    sink.write("book_odds", {"remaining": remaining, "events": data})
    return remaining


def fetch_polymarket(sink: rec.Sink) -> None:
    # end_date_min keeps zombie unresolved games (May/June, closed=false
    # forever) out of the window; without it ascending order returns only them
    min_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 6 * 3600))
    url = ("https://gamma-api.polymarket.com/events?closed=false&limit=100"
           f"&tag_slug=mlb&order=endDate&ascending=true&end_date_min={min_end}")
    events, _ = _get(url)
    games, token_ids = [], []
    # window on the game date from the slug — gamma endDate is game + ~7d
    # resolution buffer, useless for "is this game near now"
    lo = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400))
    hi = time.strftime("%Y-%m-%d", time.gmtime(time.time() + 2 * 86400))
    for e in events:
        slug = e.get("slug") or ""
        m = GAME_RE.match(slug)
        if not m or not lo <= m.group(1) <= hi:
            continue
        for mk in e.get("markets", []):
            ids = mk.get("clobTokenIds")
            if isinstance(ids, str):
                ids = json.loads(ids)
            games.append({
                "slug": slug, "question": mk.get("question"),
                "outcomes": mk.get("outcomes"), "end": e.get("endDate"),
                "bestBid": mk.get("bestBid"), "bestAsk": mk.get("bestAsk"),
                "spread": mk.get("spread"), "volume": mk.get("volume"),
                "tokens": ids})
            token_ids += ids or []
    sink.write("poly_games", games)
    for i in range(0, len(token_ids), 100):
        chunk = token_ids[i:i + 100]
        req = urllib.request.Request(
            "https://clob.polymarket.com/books",
            data=json.dumps([{"token_id": t} for t in chunk]).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            sink.write("poly_books", json.load(r))


def main() -> None:
    cycles = None
    if "--probe" in sys.argv:
        cycles = int(sys.argv[sys.argv.index("--probe") + 1])
    sink = rec.Sink("odds", out=ODDS_DIR)
    n = 0
    while cycles is None or n < cycles:
        t0 = time.time()
        remaining = None
        try:
            fetch_polymarket(sink)
        except Exception as e:  # noqa: BLE001
            sink.write("poly_err", str(e))
        try:
            remaining = fetch_odds(sink)
            if remaining is not None and remaining < RESERVE:
                sink.write("odds_stop", {"remaining": remaining})
                print(f"quota reserve hit ({remaining}), odds calls stop",
                      flush=True)
                cycles = n + 1  # finish this cycle and exit
        except Exception as e:  # noqa: BLE001
            sink.write("odds_err", str(e))
        sink.flush()
        n += 1
        if cycles is not None and n >= cycles:
            break
        time.sleep(max(0.0, POLL_MIN * 60 - (time.time() - t0)))
    if sink.fh:
        sink.fh.close()
    if "--probe" in sys.argv:
        print(f"probe done, odds quota remaining: {remaining}", flush=True)


if __name__ == "__main__":
    main()
