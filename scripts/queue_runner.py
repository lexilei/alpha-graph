"""Machine-level serial experiment queue.

Layout (~/Lex/queue/):
  pending/   job specs, plain shell scripts, run in lexicographic order
             (prefix with 010_, 020_ ... to order). Optional header
             directives in the first lines:
               # after: 2026-07-26      gate: don't start before this UTC date
               # after: /path/to/file   gate: don't start until this exists
               # mem_gb: 8              gate: need this much free+inactive RAM
  running/   the one currently-executing spec (claimed atomically by rename)
  done/      finished specs
  logs/      one log per job
  HISTORY.log  start/finish lines with exit codes

One job at a time, nice -19 + caffeinate, so the live pipeline keeps
priority and the machine stays awake only while a job runs. A gated job
does NOT block later runnable jobs: the runner picks the first RUNNABLE
spec in order. Hardening (2026-07-24 review): flock singleton, loop-level
exception guard, SIGTERM kills the whole job process group and marks the
spec interrupted, watchdog-managed. A spec found in running/ at startup
means the runner died mid-job: it is moved to done/ marked interrupted,
never silently re-run (jobs are not assumed idempotent).

Jobs run with cwd=~/Lex/queue — a spec must `cd` to its repo or use
absolute paths (learned from a live rc=127: the watchdog starts this
runner from scripts/, so relative spec paths are meaningless).

Usage: nohup .venv/bin/python scripts/queue_runner.py >> data/logs/launchd/queue_runner.log 2>&1 &
Queue a job:  cat > ~/Lex/queue/pending/030_name.sh
"""

from __future__ import annotations

import fcntl
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

Q = Path.home() / "Lex" / "queue"
PENDING, RUNNING, DONE, LOGS = (Q / d for d in
                                ("pending", "running", "done", "logs"))
HISTORY = Q / "HISTORY.log"
POLL_SEC = 60
DEFAULT_MEM_GB = 8

_current = {"proc": None, "spec": None}  # for the SIGTERM handler


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(HISTORY, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def free_mem_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = 0
    for key in ("Pages free", "Pages inactive"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        if m:
            pages += int(m.group(1))
    return pages * page / 1e9


def gates(spec: Path) -> str | None:
    """Return None if runnable, else a short reason to wait."""
    try:
        head = spec.read_text(errors="replace").splitlines()[:10]
    except OSError:
        return "vanished"  # deleted between iterdir and here: skip
    mem_need = DEFAULT_MEM_GB
    for ln in head:
        m = re.match(r"#\s*after:\s*(\S+)", ln)
        if m:
            g = m.group(1)
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", g):
                if datetime.now(timezone.utc).strftime("%Y-%m-%d") < g:
                    return f"after {g}"
            elif not Path(g).expanduser().exists():
                return f"missing {g}"
        m = re.match(r"#\s*mem_gb:\s*(\d+)", ln)
        if m:
            mem_need = int(m.group(1))
    have = free_mem_gb()
    if have < mem_need:
        return f"mem {have:.1f}<{mem_need}GB"
    return None


def move_unique(src: Path, dest_dir: Path, prefix: str = "") -> Path:
    """Move without clobbering an existing same-named archive."""
    tgt = dest_dir / f"{prefix}{src.name}"
    k = 2
    while tgt.exists():
        tgt = dest_dir / f"{prefix}{src.name}.{k}"
        k += 1
    src.rename(tgt)
    return tgt


def _on_term(signum, frame):  # noqa: ARG001
    p, spec = _current["proc"], _current["spec"]
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)  # whole job group, no orphans
        except (ProcessLookupError, PermissionError):
            pass
    if spec is not None and spec.exists():
        try:
            move_unique(spec, DONE, "INTERRUPTED_")
            log(f"INTERRUPTED {spec.name} (runner terminated)")
        except OSError:
            pass
    raise SystemExit(0)


def run_job(spec: Path) -> None:
    try:
        claimed = spec.rename(RUNNING / spec.name) or (RUNNING / spec.name)
    except OSError:
        return  # lost the claim race or spec vanished
    logf_path = LOGS / f"{spec.stem}.log"
    log(f"START {spec.name} -> {logf_path.name}")
    t0 = time.time()
    rc = -1
    try:
        with open(logf_path, "a") as lf:
            p = subprocess.Popen(
                ["caffeinate", "-ims", "nice", "-n", "19", "bash",
                 str(claimed)],
                stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True,  # own group: killable as a unit
                cwd=str(Q))  # deterministic cwd (watchdog starts us from
            # scripts/): specs must cd or use absolute paths
            _current["proc"], _current["spec"] = p, claimed
            rc = p.wait()
    except Exception as e:  # noqa: BLE001
        log(f"RUN ERROR {spec.name}: {str(e)[:150]}")
    finally:
        _current["proc"] = _current["spec"] = None
        try:
            if claimed.exists():
                move_unique(claimed, DONE)
        except OSError as e:
            log(f"ARCHIVE ERROR {spec.name}: {str(e)[:150]}")
    log(f"DONE {spec.name} rc={rc} took={time.time()-t0:.0f}s")


def main() -> None:
    for d in (PENDING, RUNNING, DONE, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    lock_f = open(Q / ".runner.lock", "a")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another queue_runner holds the lock; exiting")
        return
    signal.signal(signal.SIGTERM, _on_term)
    for orphan in RUNNING.iterdir():
        move_unique(orphan, DONE, "INTERRUPTED_")
        log(f"INTERRUPTED {orphan.name} (runner died mid-job, NOT re-run)")
    log("queue runner up")
    waiting_msg = ""
    while True:
        try:
            specs = sorted(p for p in PENDING.iterdir()
                           if p.suffix == ".sh" and not p.name.startswith("."))
            ran = False
            reasons = []
            for spec in specs:  # first RUNNABLE spec; gated heads don't
                why = gates(spec)  # block later runnable jobs
                if why is None:
                    waiting_msg = ""
                    run_job(spec)
                    ran = True
                    break
                reasons.append(f"{spec.name}: {why}")
            if ran:
                continue  # look for the next job immediately
            if reasons and reasons[0] != waiting_msg:
                log(f"WAIT {reasons[0]}")  # log head's reason once, not per poll
                waiting_msg = reasons[0]
        except Exception as e:  # noqa: BLE001
            log(f"runner loop error: {str(e)[:200]}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
