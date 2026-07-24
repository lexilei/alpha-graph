"""Robust reader for append-mode gzip logs corrupted by mid-write kills.

Our recorders write with gzip.open(path, "at"): each hourly file holds one
or more concatenated gzip MEMBERS (one per open; sync-flushed continuously).
Standard readers (gzip.open, zcat) lose data two ways:
  1. kill mid-write -> member with a bad/absent trailer; standard readers
     stop at the first CRC error, silently dropping every member appended
     after it, though the bytes are on disk.
  2. the in-progress member of a live file has no trailer yet, though its
     sync-flushed content is recoverable.

iter_members() walks the file with zlib.decompressobj: members are
decompressed incrementally in chunks (so a bad trailer still yields
everything up to the break), and member boundaries come from unused_data —
never from scanning for the 1f 8b magic, which occurs by chance inside
compressed data (~once per 64KB) and would split a valid member in half,
losing the whole file. Only after a hard decode error does it scan forward
for the next plausible member start.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

_CHUNK = 1 << 16


def _salvage(raw: bytes, i: int, bad: int, n: int) -> bytes:
    """Redo the member at i whose chunked decode raised inside chunk
    [bad:bad+_CHUNK]. A raising decompress() call loses its own output (the
    append around it never runs), so re-run full-speed up to the failing
    chunk, then byte-wise inside it: every byte decoded before the first
    bad byte is recovered. zlib is deterministic, so the refeed matches."""
    o = zlib.decompressobj(wbits=31)
    out = []
    if bad > i:
        try:
            out.append(o.decompress(raw[i:bad]))
        except (zlib.error, OSError):  # can't happen (same bytes succeeded)
            o, out, bad = zlib.decompressobj(wbits=31), [], i
    for k in range(bad, min(bad + _CHUNK, n)):
        if o.eof:
            break
        try:
            out.append(o.decompress(raw[k:k + 1]))
        except (zlib.error, OSError):
            break
    return b"".join(out)


def iter_members(raw: bytes):
    """Yield each gzip member's decompressed bytes, best-effort on corruption."""
    i, n = 0, len(raw)
    while i < n:
        o = zlib.decompressobj(wbits=31)
        out, fed, err = [], i, False
        while fed < n and not o.eof:
            try:
                out.append(o.decompress(raw[fed:fed + _CHUNK]))
            except (zlib.error, OSError):
                err = True
                break
            fed = min(fed + _CHUNK, n)
        data = _salvage(raw, i, fed, n) if err else b"".join(out)
        if data:
            yield data
        if not err and o.eof:
            nxt = fed - len(o.unused_data)
            if nxt > i:
                i = nxt
                continue
        # hard error or truncated tail: scan for the next member candidate.
        # A false-positive magic here just fails to decode and we scan again,
        # so this converges on the next real member (or EOF) without dupes.
        j = raw.find(b"\x1f\x8b", i + 2)
        if j == -1:
            return
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
    n = 0
    for d in iter_jsonl(Path(sys.argv[1])):
        n += 1
        c[d.get("src", "?")] += 1
    print(n, "records", dict(c))
