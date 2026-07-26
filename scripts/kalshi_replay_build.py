"""Build the settlement-correct replay substrate for Kalshi BTC hourly markets.

Why
---
Nothing in this project has ever been validated against a correct Kalshi
settlement. `maker_replay_sim.py` judges a *Polymarket 5-minute up/down*
policy, not this one; and the Kalshi demo environment resolves every market
"no" (212/212, `reports/audit_maker_2026-07-26.md`), so three days of demo
P&L measured a broken simulator. This script builds the substrate that any
honest test of the FV-maker needs: recorded books, joined to the price path,
joined to the *authoritative* settlement outcome.

Sources, all already on disk or public:
  data/raw/polymarket/kalshi_*.jsonl.gz   10s orderbook snapshots + the
                                          market list (close_time, strikes)
  data/raw/polymarket/brti_*.jsonl.gz     trade prints from the BRTI
                                          constituent exchanges
  api.elections.kalshi.com (public, no auth)  the settled `result` per ticker

Settlement truth is taken from the exchange AND independently reconstructed
from the recorded constituent trades, then the two are compared. Disagreement
is reported, never silently resolved -- an unchecked settlement source is
exactly what produced the demo result.

Output
  data/reference/kalshi_market_meta.json   ticker -> strikes/close/result
  data/cache/kalshi_replay_book.parquet    one row per (ts, ticker) snapshot
  data/cache/kalshi_replay_spot.parquet    1s BRTI-proxy price path

Usage
  .venv/bin/python scripts/kalshi_replay_build.py [--limit-files N] [--no-fetch]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gz_recover import iter_jsonl  # noqa: E402

RAW = ROOT / "data" / "raw" / "polymarket"
REF = ROOT / "data" / "reference"
CACHE = ROOT / "data" / "cache"
META_F = REF / "kalshi_market_meta.json"
BOOK_F = CACHE / "kalshi_replay_book.parquet"
SPOT_F = CACHE / "kalshi_replay_spot.parquet"

KALSHI_PUBLIC = "https://api.elections.kalshi.com/trade-api/v2"
# BRTI constituents that the recorder captures. Kalshi's BTC settlement is
# the CF index; this is a proxy built from the same venues, used only to
# cross-check the exchange's own result field.
BRTI_SRC = ("coinbase", "kraken", "bitstamp", "gemini", "itbit", "lmax")


def _public_get(path: str, tries: int = 3) -> dict:
    for i in range(tries):
        req = urllib.request.Request(f"{KALSHI_PUBLIC}{path}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                time.sleep(0.6 * (i + 1))
                continue
            raise
        except OSError:
            if i < tries - 1:
                time.sleep(0.6 * (i + 1))
                continue
            raise
    raise RuntimeError(path)


# ---------------------------------------------------------------- scan books

def _touch(ob: dict) -> tuple:
    """(yes_bid, yes_bid_size, yes_ask, yes_ask_size) from an orderbook_fp.

    The levels ascend in price, so the best of each side is the last entry;
    a YES ask is the complement of the best NO bid. Same convention the
    live maker uses, so a replay fill is comparable to a live one.
    """
    fp = (ob or {}).get("orderbook_fp") or {}
    yd = fp.get("yes_dollars") or []
    nd = fp.get("no_dollars") or []
    yb = ybs = ya = yas = None
    if yd:
        yb, ybs = float(yd[-1][0]), float(yd[-1][1])
    if nd:
        ya, yas = 1.0 - float(nd[-1][0]), float(nd[-1][1])
    return yb, ybs, ya, yas


def scan_books(files) -> tuple[pd.DataFrame, dict]:
    rows, meta = [], {}
    for i, f in enumerate(files, 1):
        for r in iter_jsonl(f):
            src = r.get("src")
            msg = r.get("msg")
            if src == "kalshi_markets" and isinstance(msg, list):
                for m in msg:
                    t = m.get("ticker")
                    if t and t not in meta:
                        meta[t] = {"close_time": m.get("close_time"),
                                   "floor": m.get("floor_strike"),
                                   "cap": m.get("cap_strike")}
            elif src == "kalshi" and isinstance(msg, dict):
                t = msg.get("ticker")
                if not t:
                    continue
                yb, ybs, ya, yas = _touch(msg.get("ob"))
                if yb is None and ya is None:
                    continue          # empty book: nothing to replay against
                rows.append((r["t_local"] // 1000, t, yb, ybs, ya, yas))
        if i % 20 == 0 or i == len(files):
            print(f"  books {i}/{len(files)} files, {len(rows):,} rows",
                  flush=True)
    df = pd.DataFrame(rows, columns=["ts_ms", "ticker", "yes_bid", "yes_bid_sz",
                                     "yes_ask", "yes_ask_sz"])
    return df, meta


# ----------------------------------------------------------------- scan spot

def scan_spot(files) -> pd.DataFrame:
    """1-second last-trade path across the BRTI constituent feeds."""
    rows = []
    for i, f in enumerate(files, 1):
        for r in iter_jsonl(f):
            src = r.get("src")
            if src not in BRTI_SRC:
                continue
            m = r.get("msg") or {}
            px = None
            if "p" in m:                              # coinbase-style
                px = float(m["p"])
            elif "trades" in m:                       # kraken-style
                tr = m.get("trades") or []
                if tr:
                    px = float(tr[-1][0])
            elif "price" in m:
                px = float(m["price"])
            if px:
                rows.append((r["t_local"] // 1_000_000, src, px))
        if i % 20 == 0 or i == len(files):
            print(f"  spot {i}/{len(files)} files, {len(rows):,} prints",
                  flush=True)
    df = pd.DataFrame(rows, columns=["ts_s", "venue", "px"])
    if df.empty:
        return df
    # median across venues each second: robust to one feed printing stale
    return (df.groupby("ts_s")["px"].median().rename("spot").reset_index()
            .sort_values("ts_s"))


# ------------------------------------------------------------ settlement truth

def fetch_results(meta: dict, cached: dict) -> dict:
    """Authoritative per-ticker result, fetched one EVENT at a time.

    /markets?event_ticker=... returns every strike in an hourly event in one
    response, so ~100 requests cover four days instead of ~5,000. Tickers the
    event listing does not return fall back to the single-market endpoint.
    """
    out = dict(cached)
    need = [t for t in meta
            if out.get(t, {}).get("result") in (None, "", "unknown")]
    events = sorted({t.rsplit("-", 1)[0] for t in need})
    print(f"  fetching results: {len(need)} tickers across {len(events)} events "
          f"({len(meta) - len(need)} cached)", flush=True)

    def _store(m):
        t = m.get("ticker")
        if not t:
            return
        base = dict(meta.get(t) or out.get(t) or {})
        base.update({"result": m.get("result") or "",
                     "status": m.get("status"),
                     "floor": m.get("floor_strike", base.get("floor")),
                     "cap": m.get("cap_strike", base.get("cap")),
                     "close_time": m.get("close_time", base.get("close_time")),
                     "strike_type": m.get("strike_type")})
        out[t] = base

    for i, ev in enumerate(events, 1):
        cursor = ""
        for _page in range(10):
            try:
                d = _public_get(f"/markets?event_ticker={ev}&limit=200"
                                + (f"&cursor={cursor}" if cursor else ""))
            except Exception as e:  # noqa: BLE001
                print(f"    event {ev}: {str(e)[:90]}", flush=True)
                break
            for m in d.get("markets", []) or []:
                _store(m)
            cursor = d.get("cursor") or ""
            if not cursor:
                break
        if i % 25 == 0:
            print(f"    events {i}/{len(events)}", flush=True)
        time.sleep(0.05)

    # anything the event listing did not cover
    stragglers = [t for t in need
                  if out.get(t, {}).get("result") in (None, "", "unknown")]
    if stragglers:
        print(f"  {len(stragglers)} tickers not in their event listing; "
              f"falling back to per-market", flush=True)
        for t in stragglers:
            try:
                _store({**_public_get(f"/markets/{t}").get("market", {}),
                        "ticker": t})
            except Exception as e:  # noqa: BLE001
                out.setdefault(t, dict(meta.get(t) or {}))["fetch_err"] = str(e)[:120]
                out[t].setdefault("result", "")
            time.sleep(0.05)
    return out


def brti_settle(spot_by_s: dict, close_ts: int, window_s: int = 60):
    """Time-weighted mean of the price path over the last `window_s` before
    close -- the maker's stated model of Kalshi's settlement."""
    w = [spot_by_s[t] for t in range(close_ts - window_s + 1, close_ts + 1)
         if t in spot_by_s]
    return sum(w) / len(w) if w else None


def implied_result(meta_row: dict, s: float) -> str | None:
    floor, cap = meta_row.get("floor"), meta_row.get("cap")
    if s is None:
        return None
    if floor is not None and cap is not None:
        return "yes" if floor < s <= cap else "no"
    if floor is not None:
        return "yes" if s > floor else "no"
    if cap is not None:
        return "yes" if s <= cap else "no"
    return None


def main() -> None:
    argv = sys.argv[1:]
    limit = None
    if "--limit-files" in argv:
        limit = int(argv[argv.index("--limit-files") + 1])
    do_fetch = "--no-fetch" not in argv

    REF.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    kfiles = sorted(RAW.glob("kalshi_*.jsonl.gz"))
    bfiles = sorted(RAW.glob("brti_*.jsonl.gz"))
    if limit:
        # take the most recent N and align the two streams on the same hours,
        # else the cross-check compares disjoint windows and reports nothing
        kfiles = kfiles[-limit:]
        hours = {f.stem.split("_", 1)[1] for f in kfiles}
        bfiles = [f for f in bfiles if f.stem.split("_", 1)[1] in hours]
    print(f"kalshi files {len(kfiles)}  brti files {len(bfiles)}", flush=True)

    books, meta = scan_books(kfiles)
    print(f"book rows {len(books):,}  distinct tickers {books.ticker.nunique():,}"
          f"  market-list entries {len(meta):,}", flush=True)

    spot = scan_spot(bfiles)
    print(f"spot seconds {len(spot):,}", flush=True)

    cached = {}
    if META_F.exists():
        cached = json.loads(META_F.read_text())
    # only bother with tickers we actually have book data for
    seen = {t: meta.get(t, {}) for t in books.ticker.unique()}
    for t in seen:
        if not seen[t]:
            seen[t] = {"close_time": None, "floor": None, "cap": None}
    full = fetch_results(seen, cached) if do_fetch else {**seen, **cached}
    META_F.write_text(json.dumps(full, indent=1, sort_keys=True))
    print(f"wrote {META_F}", flush=True)

    # ---- cross-check the exchange result against the recorded price path
    spot_by_s = dict(zip(spot.ts_s, spot.spot)) if not spot.empty else {}
    checks = []
    for t, v in full.items():
        ct = v.get("close_time")
        if not ct or not v.get("result"):
            continue
        close_ts = int(pd.Timestamp(ct).timestamp())
        s = brti_settle(spot_by_s, close_ts)
        imp = implied_result(v, s)
        if imp is None:
            continue
        checks.append((t, v["result"], imp, s))
    if checks:
        chk = pd.DataFrame(checks, columns=["ticker", "exchange", "implied",
                                            "settle_px"])
        agree = (chk.exchange == chk.implied).mean()
        print(f"\nsettlement cross-check on {len(chk)} markets: "
              f"exchange vs 60s BRTI-proxy TWAP agree {agree:.3%}")
        print("  exchange result mix:", chk.exchange.value_counts().to_dict())
        bad = chk[chk.exchange != chk.implied]
        if len(bad):
            print(f"  {len(bad)} disagreements, showing 8:")
            print(bad.head(8).to_string(index=False))
    else:
        print("\nno settlement cross-check possible (no overlap)")

    # ---- attach metadata to the book table and persist
    md = pd.DataFrame([{"ticker": t, **v} for t, v in full.items()])
    keep = [c for c in ("ticker", "close_time", "floor", "cap", "result",
                        "strike_type") if c in md.columns]
    out = books.merge(md[keep], on="ticker", how="left")
    # pandas 2 infers datetime64[s] from these ISO strings, so pin the unit
    # before taking the integer view -- otherwise the epoch comes out in
    # seconds and every tau is off by a factor of 1000
    out["close_ts_ms"] = (pd.to_datetime(out.close_time, utc=True, errors="coerce")
                          .astype("datetime64[ms]").astype("int64"))
    out["tau_s"] = (out.close_ts_ms - out.ts_ms) / 1000.0
    out = out[(out.tau_s > 0) & out.result.isin(["yes", "no"])]
    out.to_parquet(BOOK_F, index=False)
    if not spot.empty:
        spot.to_parquet(SPOT_F, index=False)
    print(f"\nwrote {BOOK_F}  rows {len(out):,}  markets {out.ticker.nunique():,}"
          f"  settlements {out.groupby('ticker').close_ts_ms.first().nunique():,}")
    print(f"wrote {SPOT_F}  rows {len(spot):,}")


if __name__ == "__main__":
    main()
