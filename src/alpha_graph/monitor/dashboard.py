"""Render the monitor dashboard to a self-contained HTML file.

Everything on the page traces to a repo artifact:

  - the headline accounting (N, ceiling, family-wise bar, accepted count,
    standings) comes from the LATEST "Accounting refresh" row of
    reports/factor_preregistration.md, which the ledger's own header pins as
    authoritative;
  - the factor rows come from FACTORS.md's registry tables;
  - the cache inventory comes from parquet footers.

Nothing is recomputed into a competing number and nothing is inferred from
prose. Where the ledger attaches an adjudication to a statistic — C16's +3.33
is on record as NOT clearing the ceiling despite exceeding it — the note
travels with the number into the chart, the tooltip, and the table. The bars
are geometry; the verdict is the ledger's.

Usage:
    python -m alpha_graph.monitor            # build + open
    python -m alpha_graph.monitor --serve    # rebuild on every request
"""

from __future__ import annotations

import html
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from alpha_graph.config import PROJECT_ROOT
from alpha_graph.monitor.health import CacheReport, scan_cache
from alpha_graph.monitor.ideas import Ideas, load_ideas
from alpha_graph.monitor.ledger import Ledger, Refresh, load_ledger
from alpha_graph.monitor.registry import Registry, load_registry

OUT_PATH = PROJECT_ROOT / "reports" / "dashboard.html"

# Palette — the dataviz reference instance. Chart marks use ONE categorical
# slot (blue); it validates on both surfaces. Status hues are the reserved
# status set and appear only on chips that carry a text label, never as the
# sole carrier of meaning.
SERIES_LIGHT = "#2a78d6"
SERIES_DARK = "#3987e5"

STATUS_HUE = {
    "accepted": "#0ca30c",    # good
    "candidate": "#fab219",   # warning
    "registered": "#ec835a",  # serious
    "rejected": "#898781",    # muted ink — a tombstone is a state, not an alarm
}

_MINUS = "−"  # U+2212, matching the ledger's own typography


def _fmt_t(t: float) -> str:
    return f"+{t:.2f}" if t >= 0 else f"{_MINUS}{abs(t):.2f}"


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _git_describe() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               cwd=PROJECT_ROOT, capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return f"{sha}{' +dirty' if dirty else ''}" if sha else "not a git repo"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --------------------------------------------------------------------------- #
# SVG primitives
# --------------------------------------------------------------------------- #

def _bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """Horizontal bar: square at the baseline (left), rounded at the data end
    (right) — the mark spec. Collapses to a plain rect below the radius."""
    if w <= r:
        return f"M{x},{y} h{max(w, 0.5):.2f} v{h} h-{max(w, 0.5):.2f} Z"
    return (f"M{x},{y} H{x + w - r:.2f} A{r},{r} 0 0 1 {x + w:.2f},{y + r:.2f} "
            f"V{y + h - r:.2f} A{r},{r} 0 0 1 {x + w - r:.2f},{y + h:.2f} "
            f"H{x} Z")


def _nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """Round tick values spanning [lo, hi], aiming for `count` intervals.

    The step is the 1/2/2.5/5/10 multiple of the range's magnitude whose
    resulting interval count lands CLOSEST to `count` — picking the first step
    at or above the raw interval instead rounds the ceiling series
    (2.83→2.89, magnitude 1e-2) up to a 0.1 step and leaves the axis with a
    single tick.
    """
    if hi <= lo:
        return [lo]
    span = hi - lo
    raw = span / count
    mag = 10.0 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)),
               key=lambda s: abs(span / s - count))
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 0.5:
        if v >= lo - 1e-9:
            ticks.append(round(v, 6))
        v += step
    return ticks


# --------------------------------------------------------------------------- #
# Chart: standings vs the selection bars
# --------------------------------------------------------------------------- #

def _standings_chart(refresh: Refresh, reg: Registry) -> str:
    rows = sorted(refresh.standings, key=lambda s: s.abs_t, reverse=True)
    if not rows:
        return '<p class="empty">No standings in the latest accounting row.</p>'

    W, BAND, BAR = 780, 54, 20
    ML, MR, MT, MB = 132, 64, 46, 34
    H = MT + BAND * len(rows) + MB
    plot_w = W - ML - MR

    bars = [refresh.ceiling] + ([refresh.familywise] if refresh.familywise else [])
    xmax = max(max(s.abs_t for s in rows), *bars) * 1.14

    def sx(v: float) -> float:
        return ML + (v / xmax) * plot_w

    p: list[str] = [
        f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
        f'aria-label="Absolute t-statistic of each factor on the standings, '
        f'against the selection ceiling and the family-wise threshold.">'
    ]

    # x grid + axis (solid hairlines, recessive)
    for t in _nice_ticks(0, xmax, 4):
        if t > xmax:      # a round step can overshoot the domain
            continue
        x = sx(t)
        p.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT + BAND * len(rows)}" '
                 f'class="grid"/>')
        p.append(f'<text x="{x:.1f}" y="{MT + BAND * len(rows) + 20}" '
                 f'class="tick mid">{t:g}</text>')

    # Threshold rules. Dashed *because* they are thresholds — the one place
    # dashing carries meaning rather than noise.
    def rule(v: float, label: str, dy: int) -> None:
        x = sx(v)
        p.append(f'<line x1="{x:.1f}" y1="{MT - 6}" x2="{x:.1f}" '
                 f'y2="{MT + BAND * len(rows)}" class="rule"/>')
        p.append(f'<text x="{x:.1f}" y="{dy}" class="rule-label mid">{_esc(label)}</text>')

    rule(refresh.ceiling, f"E[max] ceiling {refresh.ceiling:.2f}", MT - 26)
    if refresh.familywise:
        rule(refresh.familywise, f"5% family-wise {refresh.familywise:.2f}", MT - 10)

    # baseline at zero
    p.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + BAND * len(rows)}" '
             f'class="baseline"/>')

    for i, s in enumerate(rows):
        y = MT + i * BAND + (BAND - BAR) / 2
        w = sx(s.abs_t) - ML
        f = reg.by_id(s.factor_id)
        name = f.name if f else ""
        status = f.status if f and f.status else "—"
        tip = (f"{s.factor_id} {name} · |t| {s.abs_t:.2f} (signed {_fmt_t(s.t)})"
               + (f" · {s.note}" if s.note else "")
               + f" · status: {status}")
        p.append(f'<g class="bar-g" tabindex="0" data-tip="{_esc(tip)}">')
        p.append(f'<rect x="{ML}" y="{MT + i * BAND}" width="{plot_w + MR - 8}" '
                 f'height="{BAND}" class="hit"/>')
        p.append(f'<path d="{_bar_path(ML, y, w, BAR)}" class="bar"/>')
        p.append(f'<text x="{ML - 12}" y="{y + BAR / 2 + 1}" class="cat end">'
                 f'{_esc(s.factor_id)}</text>')
        p.append(f'<text x="{ML - 12}" y="{y + BAR / 2 + 13}" class="cat-sub end">'
                 f'{_esc(name)}</text>')
        # value at the tip, signed so the sign the |t| bar drops stays readable
        p.append(f'<text x="{sx(s.abs_t) + 10:.1f}" y="{y + BAR / 2 + 4}" '
                 f'class="val">{_esc(_fmt_t(s.t))}</text>')
        p.append('</g>')

    p.append('</svg>')

    notes = [s for s in rows if s.note]
    if notes:
        p.append('<dl class="notes">')
        for s in notes:
            p.append(f'<dt>{_esc(s.factor_id)}</dt><dd>{_esc(s.note)}</dd>')
        p.append('</dl>')
    return "".join(p)


# --------------------------------------------------------------------------- #
# Charts: the accounting history (small multiples, one measure each)
# --------------------------------------------------------------------------- #

def _line_chart(refreshes: list[Refresh], value, title: str, fmt: str,
                aria: str) -> str:
    W, H = 384, 196
    ML, MR, MT, MB = 46, 18, 26, 34
    plot_w, plot_h = W - ML - MR, H - MT - MB
    vals = [value(r) for r in refreshes]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.25 or max(abs(hi) * 0.02, 0.01)
    lo, hi = lo - pad, hi + pad
    n = len(refreshes)

    def sx(i: int) -> float:
        return ML + (plot_w if n == 1 else plot_w * i / (n - 1))

    def sy(v: float) -> float:
        return MT + plot_h - (v - lo) / (hi - lo) * plot_h

    p = [f'<svg viewBox="0 0 {W} {H}" class="chart mini" role="img" '
         f'aria-label="{_esc(aria)}">']
    for t in _nice_ticks(lo, hi, 3):
        y = sy(t)
        if not (MT - 1 <= y <= MT + plot_h + 1):
            continue
        p.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" class="tick end">{t:{fmt}}</text>')

    pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
    p.append(f'<polyline points="{pts}" class="line"/>')
    for i, (r, v) in enumerate(zip(refreshes, vals)):
        tip = f"{r.day:%Y-%m-%d} · N {r.n_looks} · ceiling {r.ceiling:.2f}"
        p.append(f'<g class="pt-g" tabindex="0" data-tip="{_esc(tip)}">')
        p.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="12" class="hit"/>')
        p.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="4" class="dot"/>')
        p.append('</g>')
    # label only the endpoints — the axis and tooltip carry the rest
    p.append(f'<text x="{sx(0):.1f}" y="{H - 12}" class="tick mid">'
             f'{refreshes[0].day:%b %d}</text>')
    if n > 1:
        p.append(f'<text x="{sx(n - 1):.1f}" y="{H - 12}" class="tick mid">'
                 f'{refreshes[-1].day:%b %d}</text>')
    p.append('</svg>')
    return f'<figure class="mini-fig"><figcaption>{_esc(title)}</figcaption>{"".join(p)}</figure>'


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Chart: the ideation queue
# --------------------------------------------------------------------------- #

def _queue_chart(ideas: Ideas) -> str:
    """Score-ranked backlog. The head carries a label, not a colour of its own:
    IDEAS.md rule 2 is 'take from the head', so the head is the one mark worth
    direct emphasis — and emphasis by annotation survives colourblindness."""
    rows = [i for i in ideas.queue if i.status == "backlog"]
    if not rows:
        return '<p class="empty">No backlog items in IDEAS.md.</p>'

    W, BAND, BAR = 780, 44, 18
    ML, MR, MT, MB = 168, 116, 20, 32
    H = MT + BAND * len(rows) + MB
    plot_w = W - ML - MR
    xmax = max(i.score for i in rows) * 1.06

    def sx(v: float) -> float:
        return ML + (v / xmax) * plot_w

    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
         f'aria-label="Ideation backlog ranked by score = P(true) x capacity / '
         f'cost x orthogonality.">']
    for t in _nice_ticks(0, xmax, 4):
        if t > xmax:      # a round step can overshoot the domain (0..7.63 -> 8)
            continue
        x = sx(t)
        p.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" '
                 f'y2="{MT + BAND * len(rows)}" class="grid"/>')
        p.append(f'<text x="{x:.1f}" y="{MT + BAND * len(rows) + 19}" '
                 f'class="tick mid">{t:g}</text>')
    p.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + BAND * len(rows)}" '
             f'class="baseline"/>')

    for i, idea in enumerate(rows):
        y = MT + i * BAND + (BAND - BAR) / 2
        w = sx(idea.score) - ML
        flag = "" if idea.score_ok else " · SCORE MISMATCH"
        tip = (f"{idea.id} {idea.name} · {idea.family} · score {idea.score:g} "
               f"= P {idea.p_true:g} x capacity {idea.capacity:g} / cost "
               f"{idea.cost:g} x ortho {idea.ortho:g}{flag}")
        p.append(f'<g class="bar-g" tabindex="0" data-tip="{_esc(tip)}">')
        p.append(f'<rect x="{ML}" y="{MT + i * BAND}" width="{plot_w + MR - 8}" '
                 f'height="{BAND}" class="hit"/>')
        p.append(f'<path d="{_bar_path(ML, y, w, BAR)}" class="bar"/>')
        short = idea.name if len(idea.name) <= 22 else idea.name[:21] + "…"
        p.append(f'<text x="{ML - 12}" y="{y + BAR / 2 + 4}" class="cat-q end">'
                 f'{_esc(idea.id)} <tspan class="cat-sub">{_esc(short)}</tspan></text>')
        label = f"{idea.score:g}" + ("  ← next up" if i == 0 else "")
        p.append(f'<text x="{sx(idea.score) + 10:.1f}" y="{y + BAR / 2 + 4}" '
                 f'class="val">{_esc(label)}</text>')
        p.append('</g>')
    p.append('</svg>')
    return "".join(p)


def _queue_table(ideas: Ideas) -> str:
    p = ['<table class="tbl"><thead><tr><th>ID</th><th>Name</th><th>Family</th>'
         '<th>Status</th><th class="num">P</th><th class="num">Capacity</th>'
         '<th class="num">Cost</th><th class="num">Ortho</th>'
         '<th class="num">Score</th><th>Registered content</th>'
         '</tr></thead><tbody>']
    for i in ideas.queue:
        promoted = (f'<span class="ref-tag">{_esc(i.promoted_to)}</span>'
                    if i.promoted_to else "")
        warn = ("" if i.score_ok else
                f'<div class="cell-note err">formula gives '
                f'{i.score_computed:.2f}</div>')
        p.append(
            f'<tr><td class="mono">{_esc(i.id)}</td>'
            f'<td class="mono">{_esc(i.name)}</td><td>{_esc(i.family)}</td>'
            f'<td>{_esc(i.status)}{promoted}</td>'
            f'<td class="num mono">{i.p_true:g}</td>'
            f'<td class="num mono">{i.capacity:g}</td>'
            f'<td class="num mono">{i.cost:g}</td>'
            f'<td class="num mono">{i.ortho:g}</td>'
            f'<td class="num mono">{i.score:g}{warn}</td>'
            f'<td class="def"><details><summary>{_esc(i.body[:88])}…</summary>'
            f'<div class="def-full">{_esc(i.body)}</div></details></td></tr>')
    p.append('</tbody></table>')
    return "".join(p)


def _family_table(ideas: Ideas) -> str:
    p = ['<table class="tbl"><thead><tr><th>Family</th><th>Status</th>'
         '<th>Evidence</th></tr></thead><tbody>']
    for f in ideas.families:
        p.append(f'<tr><td class="mono">{_esc(f.family)}</td>'
                 f'<td>{_esc(f.status)}</td>'
                 f'<td class="fam-ev">{_esc(f.evidence)}</td></tr>')
    p.append('</tbody></table>')
    return "".join(p)


def _status_chip(status: str | None) -> str:
    if not status:
        return '<span class="chip chip-none">control</span>'
    hue = STATUS_HUE.get(status, "#898781")
    return (f'<span class="chip" style="--chip:{hue}">'
            f'<span class="chip-dot"></span>{_esc(status)}</span>')


def _registry_table(reg: Registry, standings: dict[str, object]) -> str:
    p = ['<table class="tbl" id="reg-tbl"><thead><tr>'
         '<th>ID</th><th>Name</th><th>Source</th><th>Status</th>'
         '<th class="num">Headline |t|</th><th>Definition</th>'
         '</tr></thead><tbody>']
    for f in reg.candidates:
        st = standings.get(f.id)
        t_cell = (f'<span class="mono">{_esc(_fmt_t(st.t))}</span>' if st
                  else '<span class="dash">—</span>')
        note = f'<div class="cell-note">{_esc(st.note)}</div>' if st and st.note else ""
        p.append(
            f'<tr data-status="{_esc(f.status)}">'
            f'<td class="mono">{_esc(f.id)}</td>'
            f'<td class="mono">{_esc(f.name)}</td>'
            f'<td>{_esc(f.source)}</td>'
            f'<td>{_status_chip(f.status)}</td>'
            f'<td class="num">{t_cell}{note}</td>'
            f'<td class="def"><details><summary>{_esc(f.definition[:96])}…</summary>'
            f'<div class="def-full">{_esc(f.definition)}</div></details></td>'
            f'</tr>')
    p.append('</tbody></table>')
    return "".join(p)


def _baseline_table(reg: Registry) -> str:
    p = ['<table class="tbl"><thead><tr><th>ID</th><th>Name</th><th>Source</th>'
         '<th>Definition</th></tr></thead><tbody>']
    for f in reg.baseline:
        p.append(f'<tr><td class="mono">{_esc(f.id)}</td>'
                 f'<td class="mono">{_esc(f.name)}</td><td>{_esc(f.source)}</td>'
                 f'<td class="def"><details><summary>{_esc(f.definition[:96])}…</summary>'
                 f'<div class="def-full">{_esc(f.definition)}</div></details></td></tr>')
    p.append('</tbody></table>')
    return "".join(p)


def _cache_table(rep: CacheReport) -> str:
    ref = rep.reference
    p = ['<table class="tbl sortable" id="cache-tbl"><thead><tr>'
         '<th data-sort="s">Cache</th><th data-sort="n" class="num">Rows</th>'
         '<th data-sort="s">Span</th><th data-sort="n" class="num">MB</th>'
         '<th data-sort="n">Modified</th></tr></thead><tbody>']
    for e in sorted(rep.entries, key=lambda e: e.modified, reverse=True):
        older = ref is not None and e.path != ref.path and e.modified < ref.modified
        mark = ('<span class="ref-tag" title="the price panel every price-derived '
                'signal is built against">reference</span>' if ref and e.path == ref.path
                else '<span class="older" title="written before market_data.parquet">'
                     '·</span>' if older else "")
        span = e.date_range or '<span class="dash">—</span>'
        err = f'<div class="cell-note err">{_esc(e.error)}</div>' if e.error else ""
        p.append(
            f'<tr><td class="mono">{_esc(e.name)}{mark}{err}</td>'
            f'<td class="num mono" data-v="{e.rows}">{e.rows:,}</td>'
            f'<td class="mono span">{span}</td>'
            f'<td class="num mono" data-v="{e.size_bytes}">{e.size_mb:,.1f}</td>'
            f'<td class="mono" data-v="{e.modified.timestamp()}">'
            f'{e.modified:%Y-%m-%d %H:%M}</td></tr>')
    p.append('</tbody></table>')
    return "".join(p)


def _activity(led: Ledger, limit: int) -> str:
    p = ['<ol class="feed">']
    for r in led.recent(limit):
        p.append(f'<li><time>{r.day:%b %d}</time><div><p>{_esc(r.what[:260])}'
                 f'{"…" if len(r.what) > 260 else ""}</p>'
                 + (f'<p class="feed-res">{_esc(r.result[:180])}'
                    f'{"…" if len(r.result) > 180 else ""}</p>' if r.result else "")
                 + '</div></li>')
    p.append('</ol>')
    return "".join(p)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

@dataclass
class Context:
    registry: Registry
    ledger: Ledger
    cache: CacheReport
    ideas: Ideas
    built_at: datetime
    commit: str


def gather(deep: bool = False) -> Context:
    return Context(registry=load_registry(), ledger=load_ledger(),
                   cache=scan_cache(deep=deep), ideas=load_ideas(),
                   built_at=datetime.now(), commit=_git_describe())


def render(ctx: Context) -> str:
    reg, led, cache, ideas = ctx.registry, ctx.ledger, ctx.cache, ctx.ideas
    latest = led.latest
    counts = reg.status_counts()
    standings = {s.factor_id: s for s in (latest.standings if latest else [])}

    if latest:
        drift = latest.ceiling_drift()
        drift_ok = drift < 0.01
        check = (f'emax_null({latest.ceiling_trials}) recomputes to '
                 f'{latest.ceiling_recomputed():.4f} — matches the ledger to '
                 f'{drift:.4f}' if drift_ok else
                 f'MISMATCH: the ledger states {latest.ceiling:.2f} but '
                 f'emax_null({latest.ceiling_trials}) computes '
                 f'{latest.ceiling_recomputed():.4f} (drift {drift:.4f})')
        best = max(latest.standings, key=lambda s: s.abs_t, default=None)
        hero = str(latest.accepted if latest.accepted is not None else "—")
        tiles = [
            ("Tracked looks (N)", f"{latest.n_looks}", "row-based ledger count"),
            ("Selection ceiling", f"{latest.ceiling:.2f}",
             "E[max |t|] under the null — a mean, not a bar"),
            ("5% family-wise", f"{latest.familywise:.2f}" if latest.familywise else "—",
             "the threshold external claims are graded against"),
            ("Largest |t| on record", f"{best.abs_t:.2f}" if best else "—",
             f"{best.factor_id} · {best.note}" if best and best.note
             else (best.factor_id if best else "")),
        ]
    else:
        check, hero, tiles = "No accounting row found in the ledger.", "—", []

    tile_html = "".join(
        f'<div class="tile"><p class="tile-l">{_esc(l)}</p>'
        f'<p class="tile-v">{_esc(v)}</p><p class="tile-s">{_esc(s)}</p></div>'
        for l, v, s in tiles)

    filters = "".join(
        f'<button class="f-btn" data-f="{_esc(s)}">{_esc(s)} '
        f'<span class="f-n">{n}</span></button>'
        for s, n in sorted(counts.items(), key=lambda kv: -kv[1]))

    # The ideation queue. IDEAS.md may not exist (it postdates the registry),
    # so the whole section is conditional rather than rendering an empty shell.
    queue_section = ""
    if ideas.queue:
        head = ideas.head
        head_line = (f'Rule 2 — take from the head: <strong>{_esc(head.id)} '
                     f'{_esc(head.name)}</strong> ({_esc(head.family)}, score '
                     f'{head.score:g})' if head else
                     'No backlog item is available to take.')
        bad = ideas.mis_scored()
        warn = ("" if not bad else
                f'<p class="check bad">{len(bad)} score(s) disagree with '
                f'分 = P × 容量 ÷ 成本 × 正交: '
                f'{_esc(", ".join(i.id for i in bad))}</p>')
        hdr = ("" if ideas.header_ok else
               '<p class="check bad">IDEAS.md queue columns no longer match the '
               'expected header — values may be mis-mapped.</p>')
        infra = ("" if not ideas.infra else
                 '<p class="check">Infrastructure: ' + " · ".join(
                     f"<strong>{_esc(u.id)} {_esc(u.name)}</strong> [{_esc(u.status)}]"
                     for u in ideas.infra) + '</p>')
        queue_section = f"""<section>
  <h2>Ideation queue</h2>
  <div class="card">{_queue_chart(ideas)}</div>
  <p class="check">{head_line}</p>
  {hdr}{warn}{infra}
  <details class="drawer"><summary>Queue detail — {len(ideas.queue)} ideas, six fields each</summary>
    <div class="card scroll">{_queue_table(ideas)}</div>
  </details>
</section>

<section>
  <h2>Family coverage</h2>
  <div class="card scroll">{_family_table(ideas)}</div>
</section>"""

    mini = ""
    if len(led.refreshes) > 1:
        mini = (_line_chart(led.refreshes, lambda r: r.n_looks,
                            "Tracked looks (N)", "g",
                            "Tracked look count at each accounting refresh.")
                + _line_chart(led.refreshes, lambda r: r.ceiling,
                              "Selection ceiling", ".2f",
                              "Selection ceiling at each accounting refresh — it "
                              "rises as the look count grows."))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>alpha-graph monitor</title>
<style>
:root {{
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --base:#c3c2b7; --bd:rgba(11,11,11,.10);
  --series:{SERIES_LIGHT};
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --base:#383835; --bd:rgba(255,255,255,.10);
    --series:{SERIES_DARK};
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --base:#383835; --bd:rgba(255,255,255,.10);
  --series:{SERIES_DARK};
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--page); color:var(--ink);
  font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1180px; margin:0 auto; padding:32px 24px 72px; }}
header {{ display:flex; justify-content:space-between; align-items:flex-start;
  gap:16px; flex-wrap:wrap; margin-bottom:28px; }}
h1 {{ font-size:19px; margin:0 0 4px; letter-spacing:-.01em; }}
.sub {{ color:var(--muted); font-size:12.5px; margin:0; }}
.mono, .tick, .val, .cat, code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.mono {{ font-size:12.5px; font-variant-numeric:tabular-nums; }}
#theme {{ background:var(--surface); color:var(--ink2); border:1px solid var(--bd);
  border-radius:8px; padding:6px 12px; cursor:pointer; font-size:12.5px; }}
#theme:hover {{ color:var(--ink); }}
section {{ margin-bottom:34px; }}
h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.07em;
  color:var(--muted); font-weight:600; margin:0 0 12px; }}
.card {{ background:var(--surface); border:1px solid var(--bd); border-radius:12px;
  padding:22px; }}
/* hero + tiles */
.hero-row {{ display:grid; grid-template-columns:minmax(210px,1fr) 2.2fr; gap:16px; }}
.hero .h-v {{ font-size:60px; line-height:1; font-weight:600; letter-spacing:-.03em;
  margin:6px 0 8px; }}
.hero .h-l {{ font-size:13px; color:var(--ink2); margin:0; }}
.hero .h-s {{ font-size:12px; color:var(--muted); margin:0; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
  background:var(--bd); border:1px solid var(--bd); border-radius:12px; overflow:hidden; }}
.tile {{ background:var(--surface); padding:16px 18px; }}
.tile-l {{ font-size:11.5px; color:var(--muted); margin:0 0 6px; }}
.tile-v {{ font-size:27px; font-weight:600; margin:0 0 5px; letter-spacing:-.02em; }}
.tile-s {{ font-size:11px; color:var(--muted); margin:0; line-height:1.4; }}
.check {{ font-size:12px; color:var(--ink2); margin:12px 0 0; padding:9px 12px;
  border:1px solid var(--bd); border-radius:8px; background:var(--page); }}
.check.bad {{ color:#d03b3b; border-color:#d03b3b; }}
/* charts */
.chart {{ width:100%; height:auto; display:block; overflow:visible; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.baseline {{ stroke:var(--base); stroke-width:1; }}
.rule {{ stroke:var(--ink2); stroke-width:1; stroke-dasharray:3 3; }}
.rule-label {{ fill:var(--ink2); font-size:10.5px; }}
.bar {{ fill:var(--series); }}
.line {{ fill:none; stroke:var(--series); stroke-width:2; stroke-linejoin:round;
  stroke-linecap:round; }}
.dot {{ fill:var(--series); stroke:var(--surface); stroke-width:2; }}
.hit {{ fill:transparent; }}
.bar-g, .pt-g {{ cursor:default; outline:none; }}
.bar-g:hover .bar, .bar-g:focus-visible .bar,
.pt-g:hover .dot, .pt-g:focus-visible .dot {{ opacity:.72; }}
.tick {{ fill:var(--muted); font-size:10.5px; font-variant-numeric:tabular-nums; }}
.cat {{ fill:var(--ink); font-size:12px; font-weight:600; }}
.cat-sub {{ fill:var(--muted); font-size:9.5px; font-family:ui-monospace,Menlo,monospace; }}
.cat-q {{ fill:var(--ink); font-size:11.5px; font-weight:600;
  font-family:ui-monospace,Menlo,monospace; }}
.fam-ev {{ color:var(--ink2); font-size:12px; max-width:520px; }}
/* A tip label can land on a threshold rule (C16's +3.33 sits between the
   2.89 and 3.59 rules). The surface-colored halo is the surface-ring spec
   applied to text: the label stays readable without a box or a nudge that
   would detach it from its bar. */
.val {{ fill:var(--ink2); font-size:12px; font-variant-numeric:tabular-nums;
  paint-order:stroke fill; stroke:var(--surface); stroke-width:4px;
  stroke-linejoin:round; }}
.mid {{ text-anchor:middle; }} .end {{ text-anchor:end; }}
.minis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; }}
.mini-fig {{ margin:0; }}
.mini-fig figcaption {{ font-size:12px; color:var(--ink2); margin-bottom:6px; }}
.notes {{ margin:16px 0 0; padding:12px 0 0; border-top:1px solid var(--bd);
  font-size:12px; display:grid; grid-template-columns:auto 1fr; gap:4px 12px; }}
.notes dt {{ font-family:ui-monospace,Menlo,monospace; font-weight:600; color:var(--ink2); }}
.notes dd {{ margin:0; color:var(--muted); }}
/* tables */
.scroll {{ overflow-x:auto; }}
.tbl {{ width:100%; border-collapse:collapse; font-size:13px; }}
.tbl th {{ text-align:left; font-size:11px; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); font-weight:600;
  border-bottom:1px solid var(--bd); padding:0 12px 8px; white-space:nowrap; }}
.tbl td {{ border-bottom:1px solid var(--bd); padding:9px 12px; vertical-align:top; }}
.tbl tr:last-child td {{ border-bottom:none; }}
.tbl .num {{ text-align:right; }}
.sortable th[data-sort] {{ cursor:pointer; user-select:none; }}
.sortable th[data-sort]:hover {{ color:var(--ink); }}
.def {{ max-width:420px; }}
.def summary {{ cursor:pointer; color:var(--muted); font-size:12px; }}
.def summary:hover {{ color:var(--ink2); }}
.def-full {{ margin-top:8px; color:var(--ink2); font-size:12.5px; line-height:1.6; }}
.span {{ white-space:nowrap; color:var(--ink2); }}
.dash {{ color:var(--muted); }}
.cell-note {{ font-size:11px; color:var(--muted); margin-top:3px; text-align:left;
  max-width:190px; }}
.cell-note.err {{ color:#d03b3b; }}
.chip {{ display:inline-flex; align-items:center; gap:6px; font-size:11.5px;
  color:var(--ink2); white-space:nowrap; }}
.chip-dot {{ width:8px; height:8px; border-radius:50%; background:var(--chip); flex:none; }}
.chip-none {{ color:var(--muted); }}
.ref-tag {{ font-size:9.5px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--muted); border:1px solid var(--bd); border-radius:4px;
  padding:1px 5px; margin-left:7px; }}
.older {{ color:var(--muted); margin-left:6px; cursor:help; }}
.filters {{ display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; }}
.f-btn {{ background:var(--surface); border:1px solid var(--bd); border-radius:999px;
  padding:5px 12px; font-size:12px; color:var(--ink2); cursor:pointer; }}
.f-btn:hover {{ color:var(--ink); }}
.f-btn.on {{ border-color:var(--series); color:var(--ink); }}
.f-n {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
/* feed */
.feed {{ list-style:none; margin:0; padding:0; }}
.feed li {{ display:grid; grid-template-columns:56px 1fr; gap:14px; padding:11px 0;
  border-bottom:1px solid var(--bd); }}
.feed li:last-child {{ border-bottom:none; }}
.feed time {{ font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums;
  padding-top:1px; }}
.feed p {{ margin:0; font-size:12.5px; color:var(--ink2); }}
.feed-res {{ color:var(--muted); margin-top:3px; font-size:12px; }}
.empty {{ color:var(--muted); font-size:13px; margin:0; }}
.drawer {{ margin-top:10px; }}
.drawer > summary {{ cursor:pointer; color:var(--ink2); font-size:12.5px;
  list-style:none; padding:7px 12px; border:1px solid var(--bd);
  border-radius:8px; background:var(--surface); display:inline-block; }}
.drawer > summary:hover {{ color:var(--ink); }}
.drawer > summary::-webkit-details-marker {{ display:none; }}
.drawer > summary::before {{ content:"▸ "; color:var(--muted); }}
.drawer[open] > summary {{ margin-bottom:10px; }}
.drawer[open] > summary::before {{ content:"▾ "; }}
.foot {{ color:var(--muted); font-size:11.5px; border-top:1px solid var(--bd);
  padding-top:16px; }}
#tip {{ position:fixed; z-index:9; max-width:330px; background:var(--surface);
  color:var(--ink); border:1px solid var(--bd); border-radius:8px; padding:8px 11px;
  font-size:12px; line-height:1.5; pointer-events:none; opacity:0;
  transition:opacity .1s; box-shadow:0 6px 22px rgba(0,0,0,.14); }}
#tip.on {{ opacity:1; }}
@media (max-width:820px) {{ .hero-row {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>alpha-graph monitor</h1>
    <p class="sub">Accounting from the latest <code>Accounting refresh</code> row of
      reports/factor_preregistration.md · registry from FACTORS.md ·
      built {ctx.built_at:%Y-%m-%d %H:%M} · commit <code>{_esc(ctx.commit)}</code></p>
  </div>
  <button id="theme" type="button">Theme</button>
</header>

<section>
  <div class="hero-row">
    <div class="card hero">
      <p class="h-l">Accepted factors</p>
      <p class="h-v">{_esc(hero)}</p>
      <p class="h-s">of {len(reg.candidates)} registered candidates
        ({_esc(", ".join(f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: -kv[1])))})</p>
    </div>
    <div class="tiles">{tile_html}</div>
  </div>
  <p class="check{'' if not latest or drift_ok else ' bad'}">{_esc(check)}</p>
</section>

<section>
  <h2>Standings against the selection bars</h2>
  <div class="card">{_standings_chart(latest, reg) if latest else '<p class="empty">—</p>'}</div>
</section>

{f'<section><h2>Accounting history</h2><div class="card"><div class="minis">{mini}</div></div></section>' if mini else ''}

{queue_section}

<section>
  <h2>Candidate registry</h2>
  <div class="filters"><button class="f-btn on" data-f="all">all <span class="f-n">{len(reg.candidates)}</span></button>{filters}</div>
  <div class="card scroll">{_registry_table(reg, standings)}</div>
</section>

<section>
  <h2>Baseline controls</h2>
  <div class="card scroll">{_baseline_table(reg)}</div>
</section>

<section>
  <h2>Recent ledger activity</h2>
  <div class="card">{_activity(led, 12)}</div>
</section>

<section>
  <h2>Data cache</h2>
  <p class="check">{len(cache.entries)} parquet · {cache.total_bytes / 1_048_576:,.0f} MB ·
    {len(cache.part_dirs)} part directories (not walked) ·
    {len(cache.failed())} unreadable ·
    {len(cache.older_than_reference())} written before market_data.parquet (a fact
    about write order — many caches carry no price dependency at all)</p>
  <details class="drawer"><summary>Inventory — {len(cache.entries)} files, sortable</summary>
    <div class="card scroll">{_cache_table(cache)}</div>
  </details>
</section>

<p class="foot">Every number on this page is read from a repo artifact; none is
recomputed into a competing value. N is the ledger's curated count — registrations,
infrastructure notes, and role=null placebo controls are excluded by hand, so a
naive row count of the ledger's {len(led.rows)} dated rows disagrees with it by
construction and is not shown as N.</p>
</div>
<div id="tip" role="tooltip"></div>
<script>
const root = document.documentElement;
document.getElementById('theme').onclick = () => {{
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const cur = root.dataset.theme || (dark ? 'dark' : 'light');
  root.dataset.theme = cur === 'dark' ? 'light' : 'dark';
}};

// Tooltip — enhances; every value is also in a table.
const tip = document.getElementById('tip');
const show = (e, el) => {{
  tip.textContent = el.dataset.tip;
  tip.classList.add('on');
  const r = tip.getBoundingClientRect();
  const x = (e.clientX ?? el.getBoundingClientRect().left) + 14;
  const y = (e.clientY ?? el.getBoundingClientRect().top) + 14;
  tip.style.left = Math.min(x, innerWidth - r.width - 12) + 'px';
  tip.style.top = Math.min(y, innerHeight - r.height - 12) + 'px';
}};
const hide = () => tip.classList.remove('on');
for (const el of document.querySelectorAll('[data-tip]')) {{
  el.addEventListener('mousemove', e => show(e, el));
  el.addEventListener('mouseleave', hide);
  el.addEventListener('focus', e => show(e, el));
  el.addEventListener('blur', hide);
}}

// Registry status filter
const rows = [...document.querySelectorAll('#reg-tbl tbody tr')];
for (const b of document.querySelectorAll('.f-btn')) {{
  b.onclick = () => {{
    document.querySelectorAll('.f-btn').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    const f = b.dataset.f;
    rows.forEach(r => r.style.display =
      (f === 'all' || r.dataset.status === f) ? '' : 'none');
  }};
}}

// Cache table sort
for (const th of document.querySelectorAll('.sortable th[data-sort]')) {{
  let asc = false;
  th.onclick = () => {{
    const tb = th.closest('table').tBodies[0];
    const i = [...th.parentNode.children].indexOf(th);
    const num = th.dataset.sort === 'n';
    asc = !asc;
    [...tb.rows].sort((a, b) => {{
      const ca = a.cells[i], cb = b.cells[i];
      const va = num ? +(ca.dataset.v ?? 0) : ca.textContent.trim();
      const vb = num ? +(cb.dataset.v ?? 0) : cb.textContent.trim();
      return (va < vb ? -1 : va > vb ? 1 : 0) * (asc ? 1 : -1);
    }}).forEach(r => tb.appendChild(r));
  }};
}}
</script>
</body>
</html>
"""


def build(out: Path | None = None, deep: bool = False) -> Path:
    out = out or OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(gather(deep=deep)), encoding="utf-8")
    return out
