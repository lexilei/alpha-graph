"""Watchdog: keep the pipeline alive without launchd.

Every CHECK_SEC, verify each managed process is running AND (for data
producers) that its output file was written recently. Restart anything dead
or stale, nohup-detached so it survives this watchdog too. Logs every action.

This is the answer to the two session-teardown blackouts: the pipeline now
self-heals within one check interval instead of waiting for a human.

Usage: nohup .venv/bin/python scripts/watchdog.py >> data/logs/launchd/watchdog.log 2>&1 &
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LOGS = ROOT / "data" / "logs" / "launchd"
PY = ROOT / ".venv" / "bin" / "python"
CHECK_SEC = 300

# name -> (script args, freshness_glob or None, max_stale_sec)
JOBS = {
    "polymarket_recorder": ("polymarket_recorder.py", "polymarket/poly_*.jsonl.gz", 300),
    "kalshi_poller": ("kalshi_poller.py", "polymarket/kalshi_*.jsonl.gz", 300),
    "odds_poller": ("odds_poller.py", "odds/odds_*.jsonl.gz", 1800),
    "brti_recorder": ("brti_recorder.py", "polymarket/brti_*.jsonl.gz", 300),
    "deribit_recorder": ("deribit_recorder.py", "deribit/deribit_*.jsonl.gz", 300),
    "deribit_kalshi_scan": ("deribit_kalshi_scan.py --loop 120", None, 0),
    "kalshi_demo_maker": ("kalshi_demo_maker.py", None, 0),
    "perp_paper": ("perp_paper.py loop", None, 0),
}
RAW = ROOT / "data" / "raw"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}] {msg}", flush=True)


def running(script_token: str) -> bool:
    out = subprocess.run(["pgrep", "-f", script_token], capture_output=True,
                         text=True)
    return bool(out.stdout.strip())


def stale(glob: str, max_stale: int) -> bool:
    files = sorted(RAW.glob(glob))
    if not files:
        return True
    age = time.time() - files[-1].stat().st_mtime
    return age > max_stale


def start(name: str, args: str) -> None:
    logf = open(LOGS / f"{name}.log", "a")
    subprocess.Popen([str(PY), *args.split()], cwd=str(SCRIPTS),
                     stdout=logf, stderr=subprocess.STDOUT,
                     start_new_session=True)
    log(f"STARTED {name}")


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log("watchdog up")
    while True:
        for name, (args, glob, max_stale) in JOBS.items():
            token = args.split()[0]
            proc_dead = not running(token)
            data_stale = glob is not None and stale(glob, max_stale)
            if proc_dead:
                log(f"{name}: process dead -> restart")
                start(name, args)
            elif data_stale:
                # process alive but output stale: kill + restart (hung feed)
                log(f"{name}: data stale -> kill+restart")
                subprocess.run(["pkill", "-9", "-f", token])
                time.sleep(2)
                start(name, args)
        time.sleep(CHECK_SEC)


if __name__ == "__main__":
    main()
