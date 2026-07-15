"""Parse IDEAS.md — the ideation queue that feeds the registry.

IDEAS.md mirrors FACTORS.md on the selection side: FACTORS.md governs "test an
idea", IDEAS.md governs "choose what to test". Its rule 2 is the one this
module exists to serve — *take from the head of the queue, not from the latest
inspiration* — which is only enforceable if the head is visible.

Columns are parsed POSITIONALLY against a checked header, because the headers
are Chinese (P / 容量 / 成本 / 正交 / 分) and a reordered table should fail
loudly rather than silently map cost into capacity.

The stated score is cross-checked against the ledger's own formula

    分 = P(真) × 容量 ÷ 成本 × 正交

so a score left stale by rule 5 ("family 证据出来后更新分数") is visible rather
than trusted.

Usage:
    from alpha_graph.monitor.ideas import load_ideas
    ideas = load_ideas()
    ideas.queue[0]          # the head — what rule 2 says to take next
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from alpha_graph.config import PROJECT_ROOT

IDEAS_PATH = PROJECT_ROOT / "IDEAS.md"

_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
_SEPARATOR_ROW = re.compile(r"^[\s|:-]+$")
_HEADING = re.compile(r"^##\s+(.*)$")

# Section headings, matched on their leading ASCII/CJK token.
QUEUE_HEADING = "队列"
INFRA_HEADING = "基建"
FAMILY_HEADING = "Family"
INBOX_HEADING = "Inbox"

QUEUE_HEADER = ["ID", "name", "family", "status", "P", "容量", "成本", "正交", "分"]
SCORE_TOLERANCE = 0.05  # the ledger rounds to 1dp (I8: 2.475 -> 2.5)


@dataclass(frozen=True)
class Idea:
    id: str
    name: str
    family: str
    status: str
    p_true: float
    capacity: float
    cost: float
    ortho: float
    score: float
    body: str

    @property
    def score_computed(self) -> float:
        """分 = P × 容量 ÷ 成本 × 正交."""
        return self.p_true * self.capacity / self.cost * self.ortho

    @property
    def score_ok(self) -> bool:
        return abs(self.score_computed - self.score) <= SCORE_TOLERANCE

    @property
    def promoted_to(self) -> str | None:
        """The C-ID in a `promoted(C17)` status, if any. Re-uppercased: status
        is stored lower-cased for comparison, which would otherwise yield a
        `c17` that never matches a registry lookup."""
        m = re.search(r"promoted\s*\(\s*(C\d+)\s*\)", self.status, re.IGNORECASE)
        return m.group(1).upper() if m else None


@dataclass(frozen=True)
class Infra:
    id: str
    name: str
    status: str
    body: str


@dataclass(frozen=True)
class FamilyRow:
    """A row of the family coverage map. `status` is free prose and is carried
    verbatim — a keyword match on it would read "研究上有真效应,live 关闭"
    (the 8-K family: real research effect, only the LIVE path closed, C11 still
    a candidate) as a flat "closed", which is the opposite of the record."""

    family: str
    status: str
    evidence: str


@dataclass(frozen=True)
class Ideas:
    queue: list[Idea]          # score-descending
    infra: list[Infra]
    families: list[FamilyRow]
    source_path: Path
    header_ok: bool = True

    @property
    def head(self) -> Idea | None:
        """What rule 2 says to take next: the highest-scoring backlog item."""
        return next((i for i in self.queue if i.status.lower() == "backlog"), None)

    def mis_scored(self) -> list[Idea]:
        return [i for i in self.queue if not i.score_ok]


def _split_row(line: str) -> list[str]:
    parts = _UNESCAPED_PIPE.split(line.strip())
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [p.strip().replace(r"\|", "|") for p in parts]


def _num(cell: str) -> float:
    return float(re.sub(r"[^\d.]", "", cell) or "nan")


def _clean(cell: str) -> str:
    """Flatten bold and drop code ticks anywhere in the cell — I8's name is
    "`employer_reviews` / activity-nowcast 泛化", where a trailing-only strip
    leaves the inner tick behind."""
    return re.sub(r"\*\*(.+?)\*\*", r"\1", cell).replace("`", "").strip()


def parse_ideas_md(text: str, source_path: Path | None = None) -> Ideas:
    queue: list[Idea] = []
    infra: list[Infra] = []
    families: list[FamilyRow] = []
    section: str | None = None
    header_ok = True

    for line in text.splitlines():
        h = _HEADING.match(line)
        if h:
            title = h.group(1)
            if title.startswith(QUEUE_HEADING):
                section = "queue"
            elif title.startswith(INFRA_HEADING):
                section = "infra"
            elif title.startswith(FAMILY_HEADING):
                section = "family"
            elif title.startswith(INBOX_HEADING):
                section = None
            else:
                section = None
            continue

        if section is None or not line.lstrip().startswith("|"):
            continue
        if _SEPARATOR_ROW.match(line):
            continue
        cells = _split_row(line)
        if not cells:
            continue

        if section == "queue":
            if cells[0] == "ID":
                header_ok = cells[:len(QUEUE_HEADER)] == QUEUE_HEADER
                continue
            if len(cells) < 10:
                continue
            try:
                queue.append(Idea(
                    id=cells[0], name=_clean(cells[1]), family=cells[2],
                    status=cells[3].lower(), p_true=_num(cells[4]),
                    capacity=_num(cells[5]), cost=_num(cells[6]),
                    ortho=_num(cells[7]), score=_num(cells[8]), body=cells[9],
                ))
            except ValueError:
                continue
        elif section == "infra":
            if cells[0] == "ID" or len(cells) < 4:
                continue
            infra.append(Infra(id=cells[0], name=_clean(cells[1]),
                               status=cells[2].lower(), body=cells[3]))
        elif section == "family":
            if cells[0] == "family" or len(cells) < 3:
                continue
            families.append(FamilyRow(family=_clean(cells[0]),
                                      status=_clean(cells[1]),
                                      evidence=_clean(cells[2])))

    queue.sort(key=lambda i: i.score, reverse=True)
    return Ideas(queue=queue, infra=infra, families=families,
                 source_path=source_path or IDEAS_PATH, header_ok=header_ok)


def load_ideas(path: Path | None = None) -> Ideas:
    path = path or IDEAS_PATH
    if not path.exists():
        return Ideas(queue=[], infra=[], families=[], source_path=path)
    return parse_ideas_md(path.read_text(encoding="utf-8"), source_path=path)
