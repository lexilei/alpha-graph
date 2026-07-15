"""Parse reports/factor_preregistration.md — the multiple-testing ledger.

The ledger's own header pins where the truth lives:

    "the authoritative count lives in the LATEST 'Accounting refresh' table row"

so that is the only place this module reads N, the selection ceiling, and the
standings from. It deliberately does NOT recount looks from the ledger rows:
N is a curated count (registrations, infrastructure notes, and role=null
placebo controls are excluded by hand), a naive row count disagrees with it,
and in a project whose entire discipline is that number, a second
silently-different count is worse than no count.

The ceiling the refresh row states is cross-checked against a live
`ic_tools.emax_null` so drift between the prose and the code is visible rather
than assumed away.

Usage:
    from alpha_graph.monitor.ledger import load_ledger
    led = load_ledger()
    led.latest.n_looks, led.latest.ceiling, led.latest.standings
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from alpha_graph.config import PROJECT_ROOT
from alpha_graph.eval.ic_tools import emax_null

LEDGER_PATH = PROJECT_ROOT / "reports" / "factor_preregistration.md"

_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
_SEPARATOR_ROW = re.compile(r"^[\s|:-]+$")
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

_REFRESH = re.compile(r"Accounting refresh", re.IGNORECASE)
_N_LOOKS = re.compile(r"N\s*=\s*(\d+)")
_CEILING = re.compile(r"emax_null\((\d+)\)\s*=\s*([\d.]+)")
_FAMILYWISE = re.compile(r"family-wise threshold\s*[≈~=]\s*([\d.]+)")
_ACCEPTED = re.compile(r"(\d+)\s+accepted")
_STANDINGS_AT = re.compile(r"Standings[^:]*:", re.IGNORECASE)
# A factor id followed by a signed t. The ledger writes minus as U+2212.
_STANDING = re.compile(r"\b(C\d+)\s+([+\-−]\d+\.\d+)\s*(?:\(([^)]*)\))?")

_MINUS = "−"


def _to_float(signed: str) -> float:
    return float(signed.replace(_MINUS, "-"))


def _split_row(line: str) -> list[str]:
    parts = _UNESCAPED_PIPE.split(line.strip())
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [p.strip().replace(r"\|", "|") for p in parts]


def _strip_md(text: str) -> str:
    """Flatten inline markdown so a cell reads as plain prose."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip()


@dataclass(frozen=True)
class Standing:
    """One factor's headline |t| as the ledger's accounting row states it.

    `note` is the row's own parenthetical, verbatim — it is what records that
    C16's +3.33 is NOT treated as clearing the ceiling despite exceeding it.
    Any consumer that renders `t` must render `note` with it.
    """

    factor_id: str
    t: float
    note: str | None = None

    @property
    def abs_t(self) -> float:
        return abs(self.t)


@dataclass(frozen=True)
class Refresh:
    """One 'Accounting refresh' row."""

    day: date
    n_looks: int
    ceiling: float
    ceiling_trials: int | None
    familywise: float | None
    accepted: int | None
    standings: list[Standing] = field(default_factory=list)
    text: str = ""

    def ceiling_recomputed(self) -> float:
        """`emax_null` on the row's own stated trial count — the cross-check."""
        if self.ceiling_trials is None:
            return emax_null(2 * self.n_looks)
        return emax_null(self.ceiling_trials)

    def ceiling_drift(self) -> float:
        return abs(self.ceiling_recomputed() - self.ceiling)


@dataclass(frozen=True)
class LedgerRow:
    day: date
    what: str
    result: str


@dataclass(frozen=True)
class Ledger:
    refreshes: list[Refresh]
    rows: list[LedgerRow]
    source_path: Path

    @property
    def latest(self) -> Refresh | None:
        """The authoritative accounting row (the ledger header's rule)."""
        return self.refreshes[-1] if self.refreshes else None

    def recent(self, limit: int = 12) -> list[LedgerRow]:
        return self.rows[-limit:][::-1]


def parse_ledger(text: str, source_path: Path | None = None) -> Ledger:
    refreshes: list[Refresh] = []
    rows: list[LedgerRow] = []

    for line in text.splitlines():
        if not line.lstrip().startswith("|") or _SEPARATOR_ROW.match(line):
            continue
        cells = _split_row(line)
        if len(cells) < 2:
            continue
        m = _DATE.match(cells[0])
        if not m:  # header rows and the prose tables above the ledger
            continue
        day = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        what = cells[1]
        result = cells[2] if len(cells) > 2 else ""
        rows.append(LedgerRow(day=day, what=_strip_md(what), result=_strip_md(result)))

        if not _REFRESH.search(what):
            continue
        body = " ".join(cells[1:])
        n = _N_LOOKS.search(body)
        c = _CEILING.search(body)
        if not (n and c):
            continue
        fw = _FAMILYWISE.search(body)
        acc = _ACCEPTED.search(body)
        refreshes.append(Refresh(
            day=day,
            n_looks=int(n.group(1)),
            ceiling=float(c.group(2)),
            ceiling_trials=int(c.group(1)),
            familywise=float(fw.group(1)) if fw else None,
            accepted=int(acc.group(1)) if acc else None,
            standings=_parse_standings(body),
            text=_strip_md(body),
        ))

    return Ledger(refreshes=refreshes, rows=rows,
                  source_path=source_path or LEDGER_PATH)


def _parse_standings(body: str) -> list[Standing]:
    """Standings from the segment after 'Standings:' — searching the whole row
    would also match factor ids in the row's narrative preamble."""
    at = _STANDINGS_AT.search(body)
    if not at:
        return []
    seen: dict[str, Standing] = {}
    for m in _STANDING.finditer(body[at.end():]):
        fid = m.group(1)
        if fid in seen:
            continue
        note = m.group(3)
        seen[fid] = Standing(
            factor_id=fid,
            t=_to_float(m.group(2)),
            note=_strip_md(note) if note else None,
        )
    return list(seen.values())


def load_ledger(path: Path | None = None) -> Ledger:
    path = path or LEDGER_PATH
    return parse_ledger(path.read_text(encoding="utf-8"), source_path=path)
