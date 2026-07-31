"""Tests for the fixture loaders (Epic 14 Phase 14.1).

Every test builds its own fixture tree under ``tmp_path`` and passes it as
``root=`` — the real ``data/fixtures/`` subdirs do not exist in the repo yet
(DECISIONS #4), and tests must never write into the repo's fixture directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals.fixtures import (
    load_judge_alignment,
    load_judge_prompt,
    load_query_fixtures,
    load_recipe_fixtures,
    save_judge_alignment,
)
from evals.models import JudgeAlignmentRecord, JudgePrompt, RecipeFixture

# --- helpers ----------------------------------------------------------------


def _write_recipe(
    root: Path,
    fixture_set: str,
    name: str,
    *,
    expected: dict[str, object] | None = None,
    notes: str | None = None,
) -> None:
    fixture_dir = root / "synthetic_recipes" / fixture_set / name
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "source.md").write_text(f"# {name}\n\nSome recipe text.\n")
    (fixture_dir / "expected.json").write_text(json.dumps(expected or {"title": name}))
    if notes is not None:
        (fixture_dir / "notes.md").write_text(notes)


def _write_queries(root: Path, fixture_set: str, queries: str, qrels: str) -> None:
    set_dir = root / "queries" / fixture_set
    set_dir.mkdir(parents=True)
    (set_dir / "queries.tsv").write_text(queries)
    (set_dir / "qrels.tsv").write_text(qrels)


# --- load_recipe_fixtures ---------------------------------------------------


def test_recipe_fixtures_absent_set_returns_empty(tmp_path: Path) -> None:
    assert load_recipe_fixtures("missing", root=tmp_path) == []


def test_recipe_fixtures_empty_set_dir_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "synthetic_recipes" / "empty").mkdir(parents=True)
    assert load_recipe_fixtures("empty", root=tmp_path) == []


def test_recipe_fixtures_single_fixture(tmp_path: Path) -> None:
    _write_recipe(tmp_path, "synthetic", "carbonara", expected={"title": "Carbonara"})
    fixtures = load_recipe_fixtures("synthetic", root=tmp_path)
    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert isinstance(fixture, RecipeFixture)
    assert fixture.name == "carbonara"
    assert fixture.source_md.startswith("# carbonara")
    assert fixture.expected == {"title": "Carbonara"}
    assert fixture.notes is None


def test_recipe_fixtures_multiple_sorted_by_name(tmp_path: Path) -> None:
    _write_recipe(tmp_path, "synthetic", "zucchini-bake", notes="tricky boundaries\n")
    _write_recipe(tmp_path, "synthetic", "apple-pie")
    fixtures = load_recipe_fixtures("synthetic", root=tmp_path)
    assert [f.name for f in fixtures] == ["apple-pie", "zucchini-bake"]
    assert fixtures[1].notes == "tricky boundaries\n"


def test_recipe_fixture_source_is_read_as_utf8(tmp_path: Path) -> None:
    # Recipe sources are prose ("sauté", "crème brûlée", "180 °C") — the loader
    # must not depend on the ambient locale encoding.
    fixture_dir = tmp_path / "synthetic_recipes" / "synthetic" / "creme-brulee"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "source.md").write_text("Sauté, then bake at 180 °C.\n", encoding="utf-8")
    (fixture_dir / "expected.json").write_text(
        json.dumps({"title": "Crème brûlée"}, ensure_ascii=False), encoding="utf-8"
    )
    fixture = load_recipe_fixtures("synthetic", root=tmp_path)[0]
    assert "Sauté" in fixture.source_md
    assert fixture.expected == {"title": "Crème brûlée"}


# --- load_query_fixtures ----------------------------------------------------


def test_query_fixtures_absent_files_yield_empty_set(tmp_path: Path) -> None:
    fixture_set = load_query_fixtures("missing", root=tmp_path)
    assert fixture_set.name == "missing"
    assert fixture_set.queries == []
    assert fixture_set.qrels == []


def test_query_fixtures_parses_rows_skipping_comments_and_blanks(tmp_path: Path) -> None:
    _write_queries(
        tmp_path,
        "golden",
        "# query_id\tquery_text\nq1\tquick weeknight pasta\n\nq2\tvegan dessert\n",
        "# query_id\tknowledge_item_id\trelevance\nq1\tki_01\t1\n\nq2\tki_02\t2\n",
    )
    fixture_set = load_query_fixtures("golden", root=tmp_path)
    assert [(q.query_id, q.query_text) for q in fixture_set.queries] == [
        ("q1", "quick weeknight pasta"),
        ("q2", "vegan dessert"),
    ]
    assert [(r.query_id, r.knowledge_item_id, r.relevance) for r in fixture_set.qrels] == [
        ("q1", "ki_01", 1),
        ("q2", "ki_02", 2),
    ]
    assert all(isinstance(r.relevance, int) for r in fixture_set.qrels)


@pytest.mark.parametrize("present", ["queries.tsv", "qrels.tsv"])
def test_query_fixtures_half_present_set_raises(tmp_path: Path, present: str) -> None:
    # Only one of the two files: Epic 16 would otherwise score every query 0.0
    # and report a total regression that is really a missing file.
    set_dir = tmp_path / "queries" / "golden"
    set_dir.mkdir(parents=True)
    (set_dir / present).write_text("q1\tki_01\t1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="incomplete query fixture set"):
        load_query_fixtures("golden", root=tmp_path)


def test_query_fixtures_one_empty_file_raises(tmp_path: Path) -> None:
    # Present but comment-only: same false-zero regression as an absent file.
    _write_queries(tmp_path, "golden", "# query_id\tquery_text\n", "q1\tki_01\t1\n")
    with pytest.raises(ValueError, match="queries.tsv has no rows"):
        load_query_fixtures("golden", root=tmp_path)


def test_query_fixtures_both_empty_files_yield_empty_set(tmp_path: Path) -> None:
    _write_queries(tmp_path, "golden", "", "")
    fixture_set = load_query_fixtures("golden", root=tmp_path)
    assert fixture_set.queries == []
    assert fixture_set.qrels == []


def test_query_fixtures_duplicate_query_id_raises(tmp_path: Path) -> None:
    # Consumers key queries by id, so a duplicate silently drops one query.
    _write_queries(tmp_path, "golden", "q1\tpasta\nq1\tvegan pasta\n", "q1\tki_01\t1\n")
    with pytest.raises(ValueError, match="duplicate query_id"):
        load_query_fixtures("golden", root=tmp_path)


def test_query_fixtures_malformed_row_raises(tmp_path: Path) -> None:
    _write_queries(tmp_path, "broken", "q1\tonly\textra\n", "")
    with pytest.raises(ValueError, match="expected 2 tab-separated fields"):
        load_query_fixtures("broken", root=tmp_path)


# --- load_judge_prompt ------------------------------------------------------


def test_judge_prompt_missing_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_judge_prompt("missing", root=tmp_path)


def test_judge_prompt_without_version_line(tmp_path: Path) -> None:
    prompts = tmp_path / "judge_prompts"
    prompts.mkdir(parents=True)
    (prompts / "extraction-judge.md").write_text("You are a strict judge.\n")
    prompt = load_judge_prompt("extraction-judge", root=tmp_path)
    assert isinstance(prompt, JudgePrompt)
    assert prompt.name == "extraction-judge"
    assert prompt.version is None
    assert prompt.text == "You are a strict judge.\n"


def test_judge_prompt_parses_leading_version_line(tmp_path: Path) -> None:
    prompts = tmp_path / "judge_prompts"
    prompts.mkdir(parents=True)
    (prompts / "extraction-judge.md").write_text(
        "---\nversion: v2\n---\n\nYou are a strict judge.\n"
    )
    prompt = load_judge_prompt("extraction-judge", root=tmp_path)
    assert prompt.version == "v2"
    assert "strict judge" in prompt.text


def test_judge_prompt_parses_version_below_other_front_matter_keys(tmp_path: Path) -> None:
    prompts = tmp_path / "judge_prompts"
    prompts.mkdir(parents=True)
    (prompts / "extraction-judge.md").write_text(
        "---\nname: extraction-judge\nversion: v3\n---\n\nYou are a strict judge.\n"
    )
    assert load_judge_prompt("extraction-judge", root=tmp_path).version == "v3"


@pytest.mark.parametrize(
    "header",
    [
        "# version: v1",  # the form Epic 15's judge prompts use
        "<!-- version: v1 -->",  # the extraction prompt template's convention
        "version: v1",
        "---\nname: summary_quality\n# version: v1\n---",
    ],
)
def test_judge_prompt_version_declaration_forms(tmp_path: Path, header: str) -> None:
    prompts = tmp_path / "judge_prompts"
    prompts.mkdir(parents=True)
    (prompts / "summary_quality.md").write_text(
        f"{header}\n\nRate the summary.\n", encoding="utf-8"
    )
    assert load_judge_prompt("summary_quality", root=tmp_path).version == "v1"


def test_judge_prompt_version_below_a_heading_in_the_header_block(tmp_path: Path) -> None:
    prompts = tmp_path / "judge_prompts"
    prompts.mkdir(parents=True)
    (prompts / "summary_quality.md").write_text(
        "# Summary quality judge\n# version: v4\n\nRate the summary.\n", encoding="utf-8"
    )
    assert load_judge_prompt("summary_quality", root=tmp_path).version == "v4"


def test_judge_prompt_bare_version_line(tmp_path: Path) -> None:
    prompts = tmp_path / "judge_prompts"
    prompts.mkdir(parents=True)
    (prompts / "extraction-judge.md").write_text("version: v1\n\nYou are a strict judge.\n")
    assert load_judge_prompt("extraction-judge", root=tmp_path).version == "v1"


def test_judge_prompt_body_version_mention_is_not_parsed(tmp_path: Path) -> None:
    prompts = tmp_path / "judge_prompts"
    prompts.mkdir(parents=True)
    (prompts / "extraction-judge.md").write_text(
        "# Extraction judge\n\nversion: not-a-declaration\n"
    )
    assert load_judge_prompt("extraction-judge", root=tmp_path).version is None


# --- judge alignment load / save --------------------------------------------


def test_judge_alignment_absent_returns_none(tmp_path: Path) -> None:
    assert load_judge_alignment("extraction-judge", "fx-001", root=tmp_path) is None


def test_judge_alignment_present_returns_record(tmp_path: Path) -> None:
    alignment_dir = tmp_path / "judge_alignment"
    alignment_dir.mkdir(parents=True)
    (alignment_dir / "fx-001.json").write_text(
        json.dumps(
            {
                "fixture_id": "fx-001",
                "human_rating": "good",
                "judge_rating": "good",
                "agreement_status": "agree",
                "run_metadata": {"judge": "extraction-judge"},
            }
        )
    )
    record = load_judge_alignment("extraction-judge", "fx-001", root=tmp_path)
    assert record is not None
    assert record.human_rating == "good"
    assert record.run_metadata == {"judge": "extraction-judge"}


def test_judge_alignment_partial_record_loads(tmp_path: Path) -> None:
    alignment_dir = tmp_path / "judge_alignment"
    alignment_dir.mkdir(parents=True)
    (alignment_dir / "fx-002.json").write_text(json.dumps({"fixture_id": "fx-002"}))
    record = load_judge_alignment("extraction-judge", "fx-002", root=tmp_path)
    assert record is not None
    assert record.human_rating is None
    assert record.run_metadata == {}


def test_save_judge_alignment_round_trips(tmp_path: Path) -> None:
    record = JudgeAlignmentRecord(
        fixture_id="fx-003",
        human_rating="acceptable",
        judge_rating="good",
        agreement_status="disagree",
        run_metadata={"judge": "extraction-judge", "run": 3},
    )
    path = save_judge_alignment(record, root=tmp_path)
    assert path == tmp_path / "judge_alignment" / "fx-003.json"
    assert path.is_file()
    loaded = load_judge_alignment("extraction-judge", "fx-003", root=tmp_path)
    assert loaded == record
