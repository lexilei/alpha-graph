"""Nightly compaction + heavy analysis batch.

Two jobs, run once per night (or on demand):
  1. compact — parse each source's raw hourly JSONL.gz for the day into a
     single analysis-ready parquet, so downstream analyses read seconds not
     minutes. Also builds a coverage manifest (per-source gap list) so every
     study can exclude blackout intervals instead of ffill-bridging them.
  2. analyses — run the standing measurement scripts and append dated lines
     to reports/nightly_metrics.jsonl (devscan settlement P&L, latency,
     width). Each wrapped so one failure doesn't abort the batch.

Compacted outputs: data/compact/{source}_{YYYYMMDD}.parquet
Manifest:          data/compact/coverage_manifest.json

Usage:
  .venv/bin/python scripts/nightly_compact.py compact   # just compaction
  .venv/bin/python scripts/nightly_compact.py all        # compact + analyses
  .venv/bin/python scripts/nightly_compact.py loop        # nightly at 06:30 UTC
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "polymarket"
DERIBIT = ROOT / "data" / "raw" / "deribit"
COMPACT = ROOT / "data" / "compact"
PY = ROOT / ".venv" / "bin" / "python"


def log(m: str) -> None:
    print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}] {m}", flush=True)


def read_gz(f: Path):
    try:
        for line in gzip.open(f, "rt"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    except (EOFError, zlib.error, gzip.BadGzipFile, OSError):
        return


def compact_binance(day: str) -> int:
    rows = []
    for f in sorted(RAW.glob(f"binance_{day}_*.jsonl.gz")):
        for d in read_gz(f):
            if d["src"] == "binance":
                m = d["msg"].get("data", {})
                if m.get("s") == "BTCUSDT":
                    rows.append((d["t_local"], float(m["p"])))
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=["t_local", "px"])
    df.to_parquet(COMPACT / f"binance_{day}.parquet", index=False)
    return len(df)


def compact_coverage(day: str) -> dict:
    """Per-source: first/last t_local and the gap list (>60s) in seconds."""
    manifest = {}
    for src, glob in (("binance", f"binance_{day}_*.jsonl.gz"),
                      ("poly", f"poly_{day}_*.jsonl.gz"),
                      ("brti", f"brti_{day}_*.jsonl.gz"),
                      ("kalshi", f"kalshi_{day}_*.jsonl.gz"),
                      ("deribit", None)):
        base = DERIBIT if src == "deribit" else RAW
        g = glob or f"deribit_{day}_*.jsonl.gz"
        ts = []
        for f in sorted(base.glob(g)):
            for d in read_gz(f):
                ts.append(d["t_local"] // 1_000_000)
        if not ts:
            manifest[src] = {"present": False}
            continue
        ts = sorted(set(ts))
        gaps = [(ts[i], ts[i + 1]) for i in range(len(ts) - 1)
                if ts[i + 1] - ts[i] > 60]
        manifest[src] = {"present": True, "start": ts[0], "end": ts[-1],
                         "n_sec": len(ts),
                         "gaps": [[a, b, b - a] for a, b in gaps]}
    return manifest


def run_compact(day: str | None = None) -> None:
    COMPACT.mkdir(parents=True, exist_ok=True)
    if day is None:
        day = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y%m%d")
    log(f"compacting {day}")
    n = compact_binance(day)
    log(f"  binance spot: {n:,} rows")
    manifest_f = COMPACT / "coverage_manifest.json"
    manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {}
    manifest[day] = compact_coverage(day)
    manifest_f.write_text(json.dumps(manifest, indent=1))
    for src, info in manifest[day].items():
        if info.get("present"):
            log(f"  {src}: {info['n_sec']}s covered, {len(info['gaps'])} gaps")
        else:
            log(f"  {src}: ABSENT")


def run_analyses() -> None:
    out = ROOT / "reports" / "nightly_metrics.jsonl"
    stamp = datetime.now(timezone.utc).isoformat()
    for name, script in (("latency", "polymarket_latency_analysis.py"),
                         ("width", "maker_width_analysis.py"),
                         ("twap", "twap_settlement_study.py")):
        try:
            r = subprocess.run([str(PY), f"scripts/{script}"], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=1800)
            tail = "\n".join(r.stdout.strip().splitlines()[-6:])
            with open(out, "a") as f:
                f.write(json.dumps({"ts": stamp, "metric": name,
                                    "ok": r.returncode == 0,
                                    "tail": tail}) + "\n")
            log(f"analysis {name}: {'ok' if r.returncode == 0 else 'FAILED'}")
        except Exception as e:  # noqa: BLE001
            log(f"analysis {name} error: {str(e)[:120]}")


def next_run() -> float:
    now = datetime.now(timezone.utc)
    run = now.replace(hour=6, minute=30, second=0, microsecond=0)
    if run <= now:
        run += timedelta(days=1)
    return run.timestamp()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "compact":
        run_compact()
    elif mode == "all":
        run_compact()
        run_analyses()
    elif mode == "loop":
        while True:
            wait = next_run() - time.time()
            log(f"sleeping {wait/3600:.1f}h until 06:30 UTC")
            time.sleep(max(60, wait))
            try:
                run_compact()
                run_analyses()
            except Exception as e:  # noqa: BLE001
                log(f"nightly error: {str(e)[:200]}")


if __name__ == "__main__":
    main()
