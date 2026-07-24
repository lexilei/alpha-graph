"""Robust reader for append-mode gzip logs corrupted by mid-write kills.

Our recorders write with gzip.open(path, "at"): each flush is a concatenated
gzip MEMBER. When a process is killed mid-write, that member gets a bad
trailer, and standard readers (gzip.open, zcat) stop at the first CRC error —
silently dropping every member appended AFTER it, though the bytes are on
disk. Two session-teardown kills made this a real, recurring data-loss path.

iter_jsonl() splits the file on the gzip magic (1f 8b) and decompresses each
member independently, skipping only the corrupt ones. Drop-in replacement for
the per-script `read_gz` guards.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path


def iter_members(raw: bytes):
    """Yield each gzip member's decompressed bytes, skipping corrupt ones."""
    i = 0
    n = len(raw)
    while i < n:
        j = raw.find(b"\x1f\x8b", i + 2)
        if j == -1:
            j = n
        chunk = raw[i:j]
        try:
            yield zlib.decompress(chunk, wbits=31)
        except (zlib.error, OSError):
            pass  # corrupt member (kill boundary) — skip, keep the rest
        i = j


def iter_jsonl(path: Path):
    raw = Path(path).read_bytes()
    for member in iter_members(raw):
        for line in member.decode("utf-8", "replace").splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    import sys
    from collections import Counter
    c = Counter()
    for d in iter_jsonl(Path(sys.argv[1])):
        c[d.get("src", "?")] += 1
    print(dict(c))
