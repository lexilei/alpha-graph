"""Watchdog: keep the pipeline alive without launchd.

Every CHECK_SEC, verify each managed process is running AND (for data
producers) that its output file was written recently. Restart what's dead.

Hard lessons baked in (2026-07-24 adversarial review + the brti incident):
  - staleness restarts are rate-limited (STALE_COOLDOWN) and skipped inside
    a post-start grace window — when the stall is upstream, a kill loop only
    interrupts the recorder's own reconnect and burns exchange handshakes;
  - kills are SIGTERM first (flush a chance to run), SIGKILL only after a
    grace, and target exact PIDs found via a python-scoped pgrep — never
    `pkill -9 -f <substring>`, which once destroyed buffered gzip data and
    can hit unrelated processes (editors, tails, debug runs);
  - the check loop never dies: every job's check is exception-wrapped;
  - a flock singleton lock prevents two watchdogs double-starting jobs into
    the same append-mode gzip files.

Usage: nohup .venv/bin/python scripts/watchdog.py >> data/logs/launchd/watchdog.log 2>&1 &
"""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LOGS = ROOT / "data" / "logs" / "launchd"
PY = ROOT / ".venv" / "bin" / "python"
CHECK_SEC = 300
STALE_COOLDOWN = 1800   # at most one staleness restart per job per 30 min
START_GRACE = 600       # no staleness judgment within 10 min of a start

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
    "nightly_compact": ("nightly_compact.py loop", None, 0),
    "queue_runner": ("queue_runner.py", None, 0),
}
RAW = ROOT / "data" / "raw"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}] {msg}", flush=True)


def pids_of(script_token: str) -> list[int]:
    # python-scoped AND token anchored as argv[1]: won't match editors/tail
    # or `python -c '...' token`-style holders. Residual: a manual debug run
    # `python <token> --probe` still matches — avoid running those while the
    # managed copy is stale.
    out = subprocess.run(
        ["pgrep", "-f", rf"python[0-9.]* (\S*/)?{script_token}( |$)"],
        capture_output=True, text=True)
    me = os.getpid()
    return [int(p) for p in out.stdout.split() if int(p) != me]


def stale(glob: str, max_stale: int) -> bool:
    files = sorted(RAW.glob(glob))
    if not files:
        return True
    age = time.time() - files[-1].stat().st_mtime
    return age > max_stale


def terminate(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 5
    while time.time() < deadline:
        if not any(_alive(p) for p in pids):
            return
        time.sleep(0.5)
    for pid in pids:
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def start(name: str, args: str) -> None:
    with open(LOGS / f"{name}.log", "a") as logf:
        subprocess.Popen([str(PY), *args.split()], cwd=str(SCRIPTS),
                         stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    log(f"STARTED {name}")


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    lock_f = open(LOGS / "watchdog.lock", "a")  # "a": never truncate
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another watchdog holds the lock; exiting")
        return
    log("watchdog up")
    last_start: dict[str, float] = {}
    last_stale_kill: dict[str, float] = {}
    first_sweep = True
    while True:
        for name, (args, glob, max_stale) in JOBS.items():
            try:
                token = args.split()[0]
                pids = pids_of(token)
                now = time.time()
                if pids and first_sweep:
                    # adopted process: grant the same post-start grace, so a
                    # watchdog (re)boot during an upstream outage doesn't
                    # kill an alive-but-stale recorder on cycle 0
                    last_start.setdefault(name, now)
                if not pids:
                    log(f"{name}: process dead -> restart")
                    start(name, args)
                    last_start[name] = now
                    continue
                if glob is None:
                    continue
                if now - last_start.get(name, 0.0) < START_GRACE:
                    continue
                if stale(glob, max_stale):
                    if now - last_stale_kill.get(name, 0.0) < STALE_COOLDOWN:
                        continue  # restarting again this soon won't help an
                        # upstream stall; give the process's own retry a shot
                    log(f"{name}: data stale -> TERM+restart")
                    terminate(pids)
                    start(name, args)
                    last_start[name] = last_stale_kill[name] = now
            except Exception as e:  # noqa: BLE001
                log(f"{name}: watchdog check error: {str(e)[:150]}")
        first_sweep = False
        time.sleep(CHECK_SEC)


if __name__ == "__main__":
    main()
