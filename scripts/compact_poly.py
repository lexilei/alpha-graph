"""Compact the raw Polymarket book stream into columnar parquet.

The raw stream (poly_YYYYMMDD_HH.jsonl.gz, ~3.5GB/day) repeats a 78-char
asset_id, a 66-char market hash and stringified numbers on every level
change. Here each UTC day becomes four zstd parquet tables in data/compact/:

  poly_changes_{day}  one row per level change: ts_local_us, ts_exch_ms,
                      tok, side (1 bid/-1 ask), price_u, size_u,
                      best_bid_u, best_ask_u   (u = micro-units x1e6)
  poly_books_{day}    exploded full snapshots: ts_local_us, snap, tok,
                      side, price_u, size_u
  poly_trades_{day}   ts_local_us, ts_exch_ms, tok, side, price_u, size_u,
                      fee_bps, tx_hash
  poly_tokens         cumulative map tok(int) -> asset_id, market (tiny,
                      shared across days, rewritten atomically)

Level-change order within (ts, token) is preserved by row order. The
orderbook `hash` field is dropped (sync checksum, no research value; raw
files remain the ground truth during their retention window).

Verification: per-type parsed-event counts are compared with written
parquet row counts and recorded in data/compact/poly_meta.json. Retention
(--retention N, default 14 days) deletes ONLY poly_* raw files whose day
is both older than N days and marked verified in the meta file.

Usage:
  .venv/bin/python scripts/compact_poly.py 20260723      # one day
  .venv/bin/python scripts/compact_poly.py backfill      # all raw days
  .venv/bin/python scripts/compact_poly.py --retention   # prune old raw
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_sys_p = str(Path(__file__).resolve().parent)
sys.path.insert(0, _sys_p)
from gz_recover import iter_jsonl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "polymarket"
COMPACT = ROOT / "data" / "compact"
META_F = COMPACT / "poly_meta.json"
TOKENS_F = COMPACT / "poly_tokens.parquet"
RETENTION_DAYS = 14

CH_SCHEMA = pa.schema([("ts_local_us", pa.int64()), ("ts_exch_ms", pa.int64()),
                       ("tok", pa.int32()), ("side", pa.int8()),
                       ("price_u", pa.int32()), ("size_u", pa.int64()),
                       ("best_bid_u", pa.int32()), ("best_ask_u", pa.int32())])
BK_SCHEMA = pa.schema([("ts_local_us", pa.int64()), ("snap", pa.int32()),
                       ("tok", pa.int32()), ("side", pa.int8()),
                       ("price_u", pa.int32()), ("size_u", pa.int64())])
TR_SCHEMA = pa.schema([("ts_local_us", pa.int64()), ("ts_exch_ms", pa.int64()),
                       ("tok", pa.int32()), ("side", pa.int8()),
                       ("price_u", pa.int32()), ("size_u", pa.int64()),
                       ("fee_bps", pa.int16()), ("tx_hash", pa.string())])

# near-monotonic int64s take delta encoding (-31% measured); low-cardinality
# columns take dictionary; the rest stay plain under zstd.
_DELTA = "DELTA_BINARY_PACKED"
ENC = {id(CH_SCHEMA): ({"ts_local_us": _DELTA, "ts_exch_ms": _DELTA},
                       ["tok", "side", "price_u", "best_bid_u", "best_ask_u"]),
       id(BK_SCHEMA): ({"ts_local_us": _DELTA, "snap": _DELTA},
                       ["tok", "side", "price_u"]),
       id(TR_SCHEMA): ({"ts_local_us": _DELTA, "ts_exch_ms": _DELTA},
                       ["tok", "side", "price_u", "fee_bps"])}


def log(m: str) -> None:
    print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}] {m}", flush=True)


def _u(x, cap: int | None = None) -> int:
    """micro-units int; -1 for missing/unparseable."""
    if x is None:
        return -1
    try:
        v = round(float(x) * 1e6)
    except (TypeError, ValueError):
        return -1
    return v if cap is None else min(v, cap)


_SIDE = {"BUY": 1, "SELL": -1, "buy": 1, "sell": -1}


class TokenMap:
    def __init__(self):
        self.tok: dict[str, int] = {}
        self.market: dict[str, str] = {}
        if TOKENS_F.exists():
            t = pq.read_table(TOKENS_F)
            for i, a, m in zip(t["tok"].to_pylist(), t["asset_id"].to_pylist(),
                               t["market"].to_pylist()):
                self.tok[a] = i
                self.market[a] = m
        self.dirty = False

    def get(self, asset_id: str, market: str = "") -> int:
        i = self.tok.get(asset_id)
        if i is None:
            i = len(self.tok)
            self.tok[asset_id] = i
            self.market[asset_id] = market
            self.dirty = True
        return i

    def save(self) -> None:
        if not self.dirty:
            return
        items = sorted(self.tok.items(), key=lambda kv: kv[1])
        t = pa.table({"tok": pa.array([i for _, i in items], pa.int32()),
                      "asset_id": [a for a, _ in items],
                      "market": [self.market.get(a, "") for a, _ in items]})
        tmp = TOKENS_F.with_suffix(".tmp")
        pq.write_table(t, tmp, compression="zstd")
        tmp.replace(TOKENS_F)


class Cols:
    """Column accumulator flushed to a ParquetWriter per raw hour file."""

    def __init__(self, schema: pa.Schema, path: Path):
        self.schema, self.rows = schema, []
        self.path, self.writer, self.n = path, None, 0

    def add(self, row: tuple) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        arrays = [pa.array([r[k] for r in self.rows], f.type)
                  for k, f in enumerate(self.schema)]
        if self.writer is None:
            enc, dic = ENC[id(self.schema)]
            self.writer = pq.ParquetWriter(self.path, self.schema,
                                           compression="zstd", version="2.6",
                                           column_encoding=enc,
                                           use_dictionary=dic)
        self.writer.write_table(pa.table(arrays, schema=self.schema))
        self.n += len(self.rows)
        self.rows = []

    def close(self) -> int:
        self.flush()
        if self.writer:
            self.writer.close()
        elif self.path.exists():
            self.path.unlink()  # no rows this day: drop stale file
        return self.n


def compact_day(day: str) -> dict:
    files = sorted(RAW.glob(f"poly_{day}_*.jsonl.gz"))
    if not files:
        return {"present": False}
    COMPACT.mkdir(parents=True, exist_ok=True)
    toks = TokenMap()
    ch = Cols(CH_SCHEMA, COMPACT / f"poly_changes_{day}.parquet")
    bk = Cols(BK_SCHEMA, COMPACT / f"poly_books_{day}.parquet")
    tr = Cols(TR_SCHEMA, COMPACT / f"poly_trades_{day}.parquet")
    parsed = {"changes": 0, "books": 0, "trades": 0, "other": 0, "raw": 0}
    snap = 0

    def one_book(ts: int, m: dict) -> None:
        nonlocal snap
        snap += 1
        t = toks.get(m.get("asset_id", "?"), m.get("market", ""))
        for side, key in ((1, "bids"), (-1, "asks")):
            for lv in m.get(key) or []:
                bk.add((ts, snap, t, side, _u(lv.get("price")),
                        _u(lv.get("size"))))
        parsed["books"] += 1

    for f in files:
        for d in iter_jsonl(f):
            if d.get("src") != "poly":
                continue
            parsed["raw"] += 1
            ts = d.get("t_local", -1)
            m = d.get("msg")
            if isinstance(m, list):                      # initial dump
                for b in m:
                    if isinstance(b, dict):
                        one_book(ts, b)
                continue
            if not isinstance(m, dict):
                continue
            et = m.get("event_type")
            if et == "price_change":
                tex = int(m.get("timestamp") or -1)
                mkt = m.get("market", "")
                for c in m.get("price_changes") or []:
                    ch.add((ts, tex, toks.get(c.get("asset_id", "?"), mkt),
                            _SIDE.get(c.get("side"), 0), _u(c.get("price")),
                            _u(c.get("size")), _u(c.get("best_bid")),
                            _u(c.get("best_ask"))))
                    parsed["changes"] += 1
            elif et == "book":
                one_book(ts, m)
            elif et == "last_trade_price":
                tr.add((ts, int(m.get("timestamp") or -1),
                        toks.get(m.get("asset_id", "?"), m.get("market", "")),
                        _SIDE.get(m.get("side"), 0), _u(m.get("price")),
                        _u(m.get("size")), int(float(m.get("fee_rate_bps") or -1)),
                        m.get("transaction_hash", "")))
                parsed["trades"] += 1
            else:
                parsed["other"] += 1
        ch.flush(); bk.flush(); tr.flush()
    written = {"changes": ch.close(), "books": bk.close(), "trades": tr.close()}
    toks.save()

    ok = written["changes"] == parsed["changes"] and written["trades"] == parsed["trades"]
    raw_mb = sum(f.stat().st_size for f in files) / 1e6
    out_mb = sum((COMPACT / f"poly_{t}_{day}.parquet").stat().st_size
                 for t in ("changes", "books", "trades")
                 if (COMPACT / f"poly_{t}_{day}.parquet").exists()) / 1e6
    info = {"present": True, "verified": ok, "parsed": parsed,
            "written": written, "raw_mb": round(raw_mb, 1),
            "parquet_mb": round(out_mb, 1),
            "ratio": round(raw_mb / out_mb, 1) if out_mb else None,
            "n_raw_files": len(files)}
    meta = json.loads(META_F.read_text()) if META_F.exists() else {}
    meta[day] = info
    META_F.write_text(json.dumps(meta, indent=1))
    log(f"{day}: {parsed['raw']:,} msgs -> ch {written['changes']:,} "
        f"bk {written['books']:,} tr {written['trades']:,} | "
        f"{raw_mb:.0f}MB -> {out_mb:.1f}MB (x{info['ratio']}) | "
        f"verified={ok}")
    return info


def retention(days: int = RETENTION_DAYS) -> None:
    meta = json.loads(META_F.read_text()) if META_F.exists() else {}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
    for f in sorted(RAW.glob("poly_????????_*.jsonl.gz")):
        day = f.name.split("_")[1]
        if day < cutoff and meta.get(day, {}).get("verified"):
            f.unlink()
            log(f"retention: deleted {f.name}")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "backfill"
    if arg == "--retention":
        retention()
    elif arg == "backfill":
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        days = sorted({f.name.split("_")[1] for f in RAW.glob("poly_????????_*.jsonl.gz")})
        meta = json.loads(META_F.read_text()) if META_F.exists() else {}
        for day in days:
            if day == today:
                continue  # still being written; tomorrow's run gets it
            if meta.get(day, {}).get("verified"):
                continue
            compact_day(day)
    else:
        compact_day(arg)


if __name__ == "__main__":
    main()
