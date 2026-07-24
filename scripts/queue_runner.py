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

One job at a time, nice -19 + caffeinate -ims, so the live pipeline keeps
priority and the machine stays awake only while a job runs. A spec found in
running/ at startup means the runner died mid-job: it is moved to done/ and
marked interrupted, never silently re-run (jobs are not assumed idempotent).

Usage: nohup .venv/bin/python scripts/queue_runner.py >> data/logs/launchd/queue_runner.log 2>&1 &
Queue a job:  cat > ~/Lex/queue/pending/030_name.sh
"""

from __future__ import annotations

import re
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


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(HISTORY, "a") as f:
        f.write(line + "\n")


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
    head = spec.read_text(errors="replace").splitlines()[:10]
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


def run_job(spec: Path) -> None:
    claimed = RUNNING / spec.name
    try:
        spec.rename(claimed)  # atomic claim
    except OSError:
        return
    logf = LOGS / f"{spec.stem}.log"
    log(f"START {spec.name} -> {logf.name}")
    t0 = time.time()
    with open(logf, "a") as lf:
        rc = subprocess.run(
            ["caffeinate", "-ims", "nice", "-n", "19", "bash", str(claimed)],
            stdout=lf, stderr=subprocess.STDOUT).returncode
    claimed.rename(DONE / spec.name)
    log(f"DONE {spec.name} rc={rc} took={time.time()-t0:.0f}s")


def main() -> None:
    for d in (PENDING, RUNNING, DONE, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    for orphan in RUNNING.iterdir():
        orphan.rename(DONE / f"INTERRUPTED_{orphan.name}")
        log(f"INTERRUPTED {orphan.name} (runner died mid-job, NOT re-run)")
    log("queue runner up")
    waiting_msg = ""
    while True:
        specs = sorted(p for p in PENDING.iterdir()
                       if p.suffix == ".sh" and not p.name.startswith("."))
        if specs:
            why = gates(specs[0])
            if why is None:
                waiting_msg = ""
                run_job(specs[0])
                continue  # next job immediately after one finishes
            if why != waiting_msg:  # log gate reason once, not every poll
                log(f"WAIT {specs[0].name}: {why}")
                waiting_msg = why
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
