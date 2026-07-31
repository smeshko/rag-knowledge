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
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
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
                source_md=(fixture_dir / "source.md").read_text(),
                expected=json.loads((fixture_dir / "expected.json").read_text()),
                notes=notes_path.read_text() if notes_path.is_file() else None,
            )
        )
    return fixtures


def load_query_fixtures(fixture_set: str, *, root: Path | None = None) -> QueryFixtureSet:
    """Load a BEIR-style queries+qrels pair; empty lists for absent files."""
    set_dir = _resolve_root(root) / "queries" / fixture_set
    queries = [
        QueryFixture(query_id=query_id, query_text=query_text)
        for query_id, query_text in _read_tsv(set_dir / "queries.tsv", columns=2)
    ]
    qrels = [
        Qrel(query_id=query_id, knowledge_item_id=item_id, relevance=int(relevance))
        for query_id, item_id, relevance in _read_tsv(set_dir / "qrels.tsv", columns=3)
    ]
    return QueryFixtureSet(name=fixture_set, queries=queries, qrels=qrels)


def load_judge_prompt(name: str, *, root: Path | None = None) -> JudgePrompt:
    """Load a named judge prompt; raises ``FileNotFoundError`` when absent.

    An optional leading ``version:`` line (bare or inside a ``---`` front-matter
    fence) is parsed into ``version``; otherwise ``version`` is ``None``.
    """
    path = _resolve_root(root) / "judge_prompts" / f"{name}.md"
    text = path.read_text()  # raises FileNotFoundError for a missing named prompt
    version: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        if stripped.lower().startswith("version:"):
            version = stripped.split(":", 1)[1].strip()
        break
    return JudgePrompt(name=name, version=version, text=text)


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
    return JudgeAlignmentRecord.model_validate_json(path.read_text())


def save_judge_alignment(record: JudgeAlignmentRecord, *, root: Path | None = None) -> Path:
    """Write a record to ``judge_alignment/<fixture_id>.json``; returns the path."""
    directory = _resolve_root(root) / "judge_alignment"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.fixture_id}.json"
    path.write_text(record.model_dump_json(indent=2) + "\n")
    return path
