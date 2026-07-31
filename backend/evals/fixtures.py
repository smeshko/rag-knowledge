"""Fixture loaders for the evaluation harness (doc 12 § 4, doc 13 topic 12).

Layout under ``data/fixtures/`` (all paths relative to ``backend/``):

    synthetic_recipes/<set>/<name>/{source.md, expected.json, notes.md?}
    queries/<set>/{queries.tsv, qrels.tsv}
    judge_prompts/<name>.md
    judge_alignment/<fixture_id>.json

Missing-input contract (DECISIONS #4): the list loaders degrade gracefully —
``load_recipe_fixtures`` returns ``[]`` and ``load_query_fixtures`` returns an
empty ``QueryFixtureSet`` when directories/files are absent. A *named* lookup is
different: ``load_judge_prompt`` raises ``FileNotFoundError`` for a missing
prompt (caller error, not the empty-dataset case), while ``load_judge_alignment``
returns ``None`` for an absent record. Loaders are pure filesystem I/O — no
``Settings``, provider, or DB imports.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evals.models import (
    JudgeAlignmentRecord,
    JudgePrompt,
    Qrel,
    QueryFixture,
    QueryFixtureSet,
    RecipeFixture,
)

__all__ = [
    "FIXTURES_ROOT",
    "load_judge_alignment",
    "load_judge_prompt",
    "load_query_fixtures",
    "load_recipe_fixtures",
    "save_judge_alignment",
]

FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def _resolve_root(root: Path | None) -> Path:
    return FIXTURES_ROOT if root is None else root


def _read_tsv(path: Path, *, columns: int) -> list[list[str]]:
    """Read a TSV file into rows, skipping blank and ``#``-commented lines.

    Enforces a strict column count per row; returns ``[]`` for an absent file.
    """
    if not path.is_file():
        return []
    rows: list[list[str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != columns:
            raise ValueError(
                f"{path}:{line_number}: expected {columns} tab-separated fields, "
                f"got {len(fields)}"
            )
        rows.append(fields)
    return rows


def load_recipe_fixtures(fixture_set: str, *, root: Path | None = None) -> list[RecipeFixture]:
    """Load all recipe fixtures of a set, sorted by name; ``[]`` if absent/empty."""
    set_dir = _resolve_root(root) / "synthetic_recipes" / fixture_set
    if not set_dir.is_dir():
        return []
    fixtures: list[RecipeFixture] = []
    for fixture_dir in sorted(set_dir.iterdir(), key=lambda p: p.name):
        if not fixture_dir.is_dir():
            continue
        notes_path = fixture_dir / "notes.md"
        fixtures.append(
            RecipeFixture(
                name=fixture_dir.name,
                source_md=(fixture_dir / "source.md").read_text(encoding="utf-8"),
                expected=json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8")),
                notes=notes_path.read_text(encoding="utf-8") if notes_path.is_file() else None,
            )
        )
    return fixtures


def load_query_fixtures(fixture_set: str, *, root: Path | None = None) -> QueryFixtureSet:
    """Load a BEIR-style queries+qrels pair; empty lists when the set is absent.

    A *half*-present set raises: retrieval metrics aggregate over the qrels
    query ids, so qrels without their queries would score every query ``0.0``
    and report a total regression that is really just a missing file — and
    queries without qrels would silently measure nothing. The same holds when a
    file is present but parses to no rows, and for a duplicated ``query_id``
    (consumers fold queries into a dict keyed by id, so a duplicate silently
    drops one query's text). Both sides empty stays the legitimate
    empty-dataset case.
    """
    set_dir = _resolve_root(root) / "queries" / fixture_set
    queries_path = set_dir / "queries.tsv"
    qrels_path = set_dir / "qrels.tsv"
    if queries_path.is_file() != qrels_path.is_file():
        missing = queries_path if qrels_path.is_file() else qrels_path
        raise FileNotFoundError(f"incomplete query fixture set {fixture_set!r}: missing {missing}")
    queries = [
        QueryFixture(query_id=query_id, query_text=query_text)
        for query_id, query_text in _read_tsv(queries_path, columns=2)
    ]
    qrels = [
        Qrel(query_id=query_id, knowledge_item_id=item_id, relevance=int(relevance))
        for query_id, item_id, relevance in _read_tsv(qrels_path, columns=3)
    ]
    if bool(queries) != bool(qrels):
        empty = "queries.tsv" if not queries else "qrels.tsv"
        raise ValueError(f"incomplete query fixture set {fixture_set!r}: {empty} has no rows")
    query_ids = [query.query_id for query in queries]
    duplicates = sorted({query_id for query_id in query_ids if query_ids.count(query_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate query_id(s) in {queries_path}: {', '.join(duplicates)}")
    return QueryFixtureSet(name=fixture_set, queries=queries, qrels=qrels)


_VERSION_LINE = re.compile(
    r"^(?:<!--\s*)?(?:#+\s*)?version\s*:\s*(?P<version>.+?)\s*(?:-->)?$",
    re.IGNORECASE,
)


def _parse_version(text: str) -> str | None:
    """Extract an optional ``version:`` declaration from a prompt's header.

    The *header* is either a ``---`` front-matter fence (where ``version:`` may
    sit alongside ``name:``/``model:``/… in any order) or, absent a fence, the
    leading block of lines up to the first blank line. Within it the declaration
    may be bare (``version: v1``), Markdown-prefixed (``# version: v1`` — the
    form Epic 15's judge prompts use), or HTML-commented
    (``<!-- version: v1 -->``, the convention of the extraction prompt template).

    The prompt *body* is never scanned, so a sentence mentioning a version deep
    in the instructions cannot be mistaken for the declaration.
    """
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines = lines[1:]
    header: list[str] = []
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            header.append(line)
    else:
        for line in lines:
            if not line.strip():
                break
            header.append(line)
    for line in header:
        match = _VERSION_LINE.match(line.strip())
        if match:
            return match.group("version").strip() or None
    return None


def load_judge_prompt(name: str, *, root: Path | None = None) -> JudgePrompt:
    """Load a named judge prompt; raises ``FileNotFoundError`` when absent.

    An optional ``version:`` declaration (bare leading line, or any key inside a
    ``---`` front-matter fence) is parsed into ``version``; otherwise ``None``.
    """
    path = _resolve_root(root) / "judge_prompts" / f"{name}.md"
    text = path.read_text(encoding="utf-8")  # raises FileNotFoundError for a missing named prompt
    return JudgePrompt(name=name, version=_parse_version(text), text=text)


def load_judge_alignment(
    name: str, fixture_id: str, *, root: Path | None = None
) -> JudgeAlignmentRecord | None:
    """Load the alignment record for a fixture, or ``None`` when absent.

    ``name`` selects the judge namespace per the epic's signature; the on-disk
    layout today is a flat ``judge_alignment/<fixture_id>.json`` (doc 13
    topic 12), so ``name`` is accepted but not yet used for path resolution.
    """
    del name  # single-file fallback layout today; per-judge namespacing is future work
    path = _resolve_root(root) / "judge_alignment" / f"{fixture_id}.json"
    if not path.is_file():
        return None
    return JudgeAlignmentRecord.model_validate_json(path.read_text(encoding="utf-8"))


def save_judge_alignment(record: JudgeAlignmentRecord, *, root: Path | None = None) -> Path:
    """Write a record to ``judge_alignment/<fixture_id>.json``; returns the path."""
    directory = _resolve_root(root) / "judge_alignment"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.fixture_id}.json"
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
