#!/usr/bin/env python3
"""Generate a single self-contained HTML viewer for the implementation epics and plans.

The Markdown is the source of truth; this script re-embeds it verbatim into the
``md-to-viewer`` blueprint template (``scripts/epics_viewer_template.html``) and writes
``docs/implementation/epics-viewer.html``. Re-run it after editing any epic or plan to
refresh the HTML.

Sources scanned:
  docs/implementation/EPICS.md          -> "Overview" doc (status table)
  docs/implementation/epics/NN-*.md     -> one doc per epic, in number order
  .claude/plans/*/                      -> active plans (PLAN + DECISIONS + RESEARCH + tasks)
  .claude/plans/archive/*/              -> archived plans (same shape)

The doc switcher holds the overview + one doc per epic. Each epic's plans are nested
*inside* that epic's doc: every plan becomes a single ``## Plan X.Y`` section inserted
right after its matching ``## Phase X.Y`` section, with the plan's PLAN/DECISIONS/
RESEARCH/tasks demoted beneath it. So plans live under their epic and never render as an
empty separate document.

No third-party dependencies; standard library only.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "scripts" / "epics_viewer_template.html"
EPICS_MD = ROOT / "docs" / "implementation" / "EPICS.md"
EPICS_DIR = ROOT / "docs" / "implementation" / "epics"
PLANS_DIR = ROOT / ".claude" / "plans"
ARCHIVE_DIR = PLANS_DIR / "archive"
OUTPUT = ROOT / "docs" / "implementation" / "epics-viewer.html"

# Stable across regenerations so reader comments (keyed by this prefix) survive a rebuild.
STORAGE_KEY = "rag-recipes-epics-v1"
EXPORT_SLUG = "epics-viewer"
EXPORT_TITLE = "rag-recipes epics & plans — review comments"

# --- dir name: [YYYY-MM-DD-]epic-<E>-phase-<MAJ>-<MIN>-<rest> -----------------------------
PLAN_DIR_RE = re.compile(
    r"^(?:(?P<date>\d{4}-\d{2}-\d{2})-)?epic-(?P<epic>\d+)-phase-(?P<maj>\d+)-(?P<min>\d+)-"
)
# A row of the EPICS.md status table: | N | [Title](link) | phases | deps | status |
TABLE_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*\[([^\]]+)\]\(([^)]+)\)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
)
HEADING_RE = re.compile(r"^(#{1,6})(\s.*)$")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def embed_safe(md: str) -> str:
    """Make Markdown safe to sit inside a <script type="text/markdown"> block."""
    return md.replace("</script>", "<\\/script>")


def attr(value: str) -> str:
    return html.escape(value, quote=True)


def demote_headings(md: str, levels: int) -> str:
    """Add ``levels`` ``#`` to every ATX heading, skipping fenced code blocks (caps at 6)."""
    if levels <= 0:
        return md
    out: list[str] = []
    in_fence = False
    for line in md.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            out.append(line)
            continue
        m = HEADING_RE.match(line)
        if m and not in_fence:
            hashes = "#" * min(6, len(m.group(1)) + levels)
            out.append(hashes + m.group(2))
        else:
            out.append(line)
    return "\n".join(out)


def neutralize_fence_headings(md: str) -> str:
    r"""Space-prefix ``#``/``##`` lines that live *inside* fenced code blocks.

    The bundled viewer's section splitter is not fence-aware: a shell comment like
    ``# Inspect health.`` inside a ```` ```bash ```` block starts with ``# `` and would be
    treated as a heading, spawning a junk nav section. A single leading space stops the
    ``^#{1,2}\s`` match while the code still renders verbatim. Real headings live outside
    fences and are untouched.
    """
    out: list[str] = []
    in_fence = False
    for line in md.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            out.append(line)
        elif in_fence and re.match(r"^#{1,2}\s", line):
            out.append(" " + line)
        else:
            out.append(line)
    return "\n".join(out)


def strip_status_line(md: str) -> str:
    """Drop a leading standalone ``**Status**: …`` line from an epic body.

    It sits between the ``#`` title and the first ``##``, which would make the viewer
    emit a synthetic duplicate "Overview" section. The status is already shown in the
    doc's sidebar sub-line, so removing it here is lossless for the reader.
    """
    lines = md.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("**Status**:"):
            del lines[i]
            break
        if line.startswith("## "):  # reached the first section; nothing to strip
            break
    return "\n".join(lines)


def strip_h1(md: str) -> str:
    """Drop the leading ``# …`` title line (and any blank lines above it)."""
    lines = md.split("\n")
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i < len(lines) and re.match(r"^# ", lines[i]):
        del lines[i]
    return "\n".join(lines).strip("\n")


def short_title(h1: str) -> str:
    """'# Plan: Epic 9 Phase 9.2 — LLM call ...' -> 'LLM call ...' (text after the em dash)."""
    text = h1.lstrip("#").strip()
    if "—" in text:
        return text.split("—", 1)[1].strip()
    return text.removeprefix("Plan:").strip()


def parse_epic_table() -> dict[int, dict]:
    """Map epic number -> {title, phases, deps, status} from the EPICS.md status table."""
    epics: dict[int, dict] = {}
    for line in read(EPICS_MD).split("\n"):
        m = TABLE_ROW_RE.match(line)
        if not m:
            continue
        num, title, _, phases, deps, status = m.groups()
        epics[int(num)] = {
            "title": title.strip(),
            "phases": phases.strip(),
            "deps": deps.strip(),
            "status": status.strip(),
        }
    return epics


def collect_plans() -> list[dict]:
    """One record per plan folder (active + archived), sorted by epic then phase."""
    plans: list[dict] = []
    dirs = [d for d in PLANS_DIR.iterdir() if d.is_dir() and d.name != "archive"]
    archived = [d for d in ARCHIVE_DIR.iterdir() if d.is_dir()] if ARCHIVE_DIR.exists() else []
    for d in dirs + archived:
        m = PLAN_DIR_RE.match(d.name)
        plan_md = d / "PLAN.md"
        if not m or not plan_md.exists():
            continue
        epic = int(m.group("epic"))
        maj, mn = int(m.group("maj")), int(m.group("min"))
        is_archived = d in archived
        plan_text = read(plan_md)
        h1 = plan_text.split("\n", 1)[0]
        plans.append(
            {
                "dir": d,
                "epic": epic,
                "maj": maj,
                "min": mn,
                "phase": f"{maj}.{mn}",
                "slug": f"p-{maj}-{mn}",
                "short": short_title(h1),
                "archived": is_archived,
                "date": m.group("date"),
                "plan_text": plan_text,
            }
        )
    plans.sort(key=lambda p: (p["epic"], p["maj"], p["min"]))
    return plans


def plan_state(plan: dict) -> str:
    if not plan["archived"]:
        return "active"
    return f"archived · {plan['date']}" if plan["date"] else "archived"


def plan_section_md(plan: dict) -> str:
    """Render one plan as a single ``## Plan X.Y`` section to nest inside its epic doc.

    All of the plan's own headings are demoted beneath this ``##`` so the whole plan stays
    in one navigable section: PLAN.md becomes h3 groups, with the decision log, research
    notes, and task breakdown grouped under their own h3 sub-headings.
    """
    d = plan["dir"]
    out = [f"## Plan {plan['phase']} — {plan['short']}  ·  {plan_state(plan)}", ""]
    # PLAN.md: drop its redundant '# Plan: …' line, demote '##' -> '###'.
    out.append(demote_headings(strip_h1(plan["plan_text"]), 1))

    decisions = d / "DECISIONS.md"
    if decisions.exists():
        out += ["", "### Decision log", "", demote_headings(strip_h1(read(decisions)), 2)]

    research = d / "RESEARCH.md"
    if research.exists():
        out += ["", "### Research notes", "", demote_headings(strip_h1(read(research)), 2)]

    tasks_dir = d / "tasks"
    if tasks_dir.is_dir():
        task_files = sorted(tasks_dir.glob("*.md"))
        if task_files:
            out += ["", "### Task breakdown", ""]
            # Demote +3 (keeping each task's '# TASK-NNN' title as an h4) so tasks sit
            # under "Task breakdown" without spawning their own nav sections.
            out += [demote_headings(read(t).strip(), 3) for t in task_files]

    return "\n".join(out)


def split_top_sections(body: str) -> list[str]:
    """Split an epic body into chunks at each ``## `` heading, ignoring fenced code."""
    chunks: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in body.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and re.match(r"^## ", line):
            chunks.append("\n".join(cur))
            cur = []
        cur.append(line)
    chunks.append("\n".join(cur))
    return chunks


def nest_plans_into_epic(epic_body: str, epic_num: int, plans: list[dict]) -> str:
    """Insert each plan as a section right after its matching ``## Phase X.Y`` section.

    Plans whose phase has no matching phase heading are appended at the end.
    """
    mine = {(p["maj"], p["min"]): p for p in plans if p["epic"] == epic_num}
    out: list[str] = []
    used: set[tuple[int, int]] = set()
    for chunk in split_top_sections(epic_body):
        out.append(chunk.rstrip())
        m = re.match(r"^## Phase (\d+)\.(\d+)\b", chunk)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            if key in mine:
                out.append(plan_section_md(mine[key]))
                used.add(key)
    for key in sorted(mine):
        if key not in used:
            out.append(plan_section_md(mine[key]))
    return "\n\n".join(c for c in out if c.strip())


def doc_block(slug: str, title: str, sub: str, file_hint: str, md: str) -> str:
    return (
        f'<script type="text/markdown" data-slug="{attr(slug)}" data-title="{attr(title)}"'
        f' data-sub="{attr(sub)}" data-file="{attr(file_hint)}">\n'
        f"{embed_safe(neutralize_fence_headings(md))}\n"
        f"</script>"
    )


def build_docs(epics_meta: dict[int, dict], plans: list[dict]) -> list[str]:
    blocks: list[str] = []

    # --- Overview (EPICS.md) ---
    blocks.append(
        doc_block(
            slug="overview",
            title="Overview",
            sub=f"{len(epics_meta)} epics · {len(plans)} plans",
            file_hint="EPICS.md",
            md=read(EPICS_MD),
        )
    )

    # --- One doc per epic, with its plans nested as sections under each phase ---
    for path in sorted(EPICS_DIR.glob("*.md")):
        prefix = re.match(r"(\d+)", path.name)
        if not prefix:
            continue
        num = int(prefix.group(1))
        meta = epics_meta.get(num, {})
        title = meta.get("title") or path.stem
        status = meta.get("status", "")
        phases = meta.get("phases", "")
        n_plans = sum(1 for p in plans if p["epic"] == num)
        sub = " · ".join(
            x
            for x in (status, f"{phases} phases" if phases else "", f"{n_plans} plans" if n_plans else "")
            if x
        )
        body = strip_status_line(read(path)).rstrip()
        md = nest_plans_into_epic(body, num, plans)
        blocks.append(
            doc_block(
                slug=f"e{num:02d}",
                title=f"Epic {num:02d} — {title}",
                sub=sub,
                file_hint=path.name,
                md=md,
            )
        )

    return blocks


def main() -> None:
    template = read(TEMPLATE)
    epics_meta = parse_epic_table()
    plans = collect_plans()
    blocks = build_docs(epics_meta, plans)

    # Fill identity placeholders.
    n_active = sum(1 for p in plans if not p["archived"])
    n_arch = sum(1 for p in plans if p["archived"])
    replacements = {
        "{{TITLE}}": "rag-recipes · Epics & Plans",
        "{{BRAND_KICK}}": "rag-recipes backend",
        "{{BRAND_TITLE}}": "Implementation<br>Epics &amp; Plans",
        "{{BRAND_SUB}}": f"{len(epics_meta)} epics · {len(plans)} plans nested",
        "{{STORAGE_KEY}}": STORAGE_KEY,
        "{{EXPORT_SLUG}}": EXPORT_SLUG,
        "{{EXPORT_TITLE}}": EXPORT_TITLE,
    }
    for needle, value in replacements.items():
        template = template.replace(needle, value)

    # Swap the two example docs (between the DOCS markers) for the real embedded docs,
    # keeping the instructional DOCS:START / DOCS:END comments intact.
    header_end = template.index("-->", template.index("DOCS:START")) + len("-->")
    if template[header_end : header_end + 1] == "\n":
        header_end += 1
    end_comment = template.rindex("<!--", 0, template.index("DOCS:END"))
    template = template[:header_end] + "\n".join(blocks) + "\n" + template[end_comment:]

    OUTPUT.write_text(template, encoding="utf-8")

    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print(f"  docs embedded : {len(blocks)} (1 overview + {len(epics_meta)} epics)")
    print(f"  plans nested  : {len(plans)} ({n_active} active, {n_arch} archived) under their epics")
    print(f"  size          : {OUTPUT.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
