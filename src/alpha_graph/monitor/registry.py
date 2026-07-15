"""Parse FACTORS.md into structured factor records.

Only the registry table's *structural* columns are parsed — ID, name, source,
status. The definition cell is carried through verbatim as prose and is never
scraped for statistics: those cells are multi-hundred-word paragraphs holding
superseded, pre-PIT, and attribution-era numbers side by side with the
headline, and a regex cannot tell which is which. Headline standings come from
the ledger's authoritative accounting row instead (see `ledger.py`).

Usage:
    from alpha_graph.monitor.registry import load_registry
    reg = load_registry()          # -> Registry(candidates=[...], baseline=[...])
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from alpha_graph.config import PROJECT_ROOT

FACTORS_PATH = PROJECT_ROOT / "FACTORS.md"

# Markdown escapes a literal pipe inside a cell as `\|` (the registry uses it
# constantly, e.g. `sn incr \|t\| >= 1.5`). Split on unescaped pipes only.
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
_SEPARATOR_ROW = re.compile(r"^[\s|:-]+$")
_HEADING = re.compile(r"^##\s+(.*)$")

CANDIDATE_HEADING = "Candidates"
BASELINE_HEADING = "Baseline"

# Statuses the registry's own legend defines, plus `registered` (used by rows
# that are pinned but not yet evaluated). Anything else is surfaced verbatim
# and flagged, rather than silently bucketed.
KNOWN_STATUSES = ("accepted", "candidate", "registered", "rejected")


@dataclass(frozen=True)
class Factor:
    """One registry row. `status` is None for baseline controls (that table has
    no status column — controls are not tested for promotion)."""

    id: str
    name: str
    source: str
    status: str | None
    definition: str
    group: str  # "candidate" | "baseline"

    @property
    def status_known(self) -> bool:
        return self.status is None or self.status in KNOWN_STATUSES


@dataclass(frozen=True)
class Registry:
    candidates: list[Factor]
    baseline: list[Factor]
    source_path: Path

    @property
    def all(self) -> list[Factor]:
        return self.candidates + self.baseline

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.candidates:
            counts[f.status or "unknown"] = counts.get(f.status or "unknown", 0) + 1
        return counts

    def by_id(self, factor_id: str) -> Factor | None:
        return next((f for f in self.all if f.id == factor_id), None)


def _split_row(line: str) -> list[str]:
    """Cells of a markdown table row, unescaping `\\|` back to `|`."""
    parts = _UNESCAPED_PIPE.split(line.strip())
    # A well-formed row starts and ends with a pipe -> empty first/last parts.
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [p.strip().replace(r"\|", "|") for p in parts]


def _strip_code_ticks(cell: str) -> str:
    return cell.strip().strip("`").strip()


def parse_factors_md(text: str, source_path: Path | None = None) -> Registry:
    """Parse the candidate and baseline tables out of FACTORS.md.

    Rows are attributed to a table by the `##` heading they sit under, so
    prose, the "Known issues" list, and the "Convention" section are ignored
    without needing to match their content.
    """
    candidates: list[Factor] = []
    baseline: list[Factor] = []
    section: str | None = None

    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            title = heading.group(1)
            if title.startswith(CANDIDATE_HEADING):
                section = "candidate"
            elif title.startswith(BASELINE_HEADING):
                section = "baseline"
            else:
                section = None
            continue

        if section is None or not line.lstrip().startswith("|"):
            continue
        if _SEPARATOR_ROW.match(line):
            continue

        cells = _split_row(line)
        if not cells or cells[0] == "ID":  # header row
            continue

        if section == "candidate":
            if len(cells) < 5:
                continue
            candidates.append(Factor(
                id=cells[0],
                name=_strip_code_ticks(cells[1]),
                source=cells[2],
                status=cells[3].lower(),
                definition=cells[4],
                group="candidate",
            ))
        else:
            if len(cells) < 4:
                continue
            baseline.append(Factor(
                id=cells[0],
                name=_strip_code_ticks(cells[1]),
                source=cells[2],
                status=None,
                definition=cells[3],
                group="baseline",
            ))

    return Registry(candidates=candidates, baseline=baseline,
                    source_path=source_path or FACTORS_PATH)


def load_registry(path: Path | None = None) -> Registry:
    path = path or FACTORS_PATH
    return parse_factors_md(path.read_text(encoding="utf-8"), source_path=path)
